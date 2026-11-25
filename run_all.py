#!/usr/bin/env python3
"""One command for the whole study: setup -> data -> preprocessing -> lesion pretraining ->
training (several seeds, including the decoupled-head stage) -> evaluation -> external cohort ->
screening -> XAI against expert masks -> label-noise pass -> baselines/ablations -> final
statistics -> parameter log -> PDF report.

    python run_all.py                  # the real study (needs a CUDA GPU and credentials, see below)
    python run_all.py --smoke          # synthetic data, tiny model, CPU, minutes: verifies every step
    python run_all.py --preflight-only # checks credentials / GPU / disk / network and stops

Credentials, as environment variables (nothing is ever written to disk or logged):
    KAGGLE_USERNAME, KAGGLE_KEY   EyePACS and DDR download (Kaggle account -> Create New API Token)
    HF_TOKEN                      RETFound weights (gated repo: request access once on HuggingFace)

Behaviour designed for unattended cloud runs
  * Resumable: state is kept in <run>/run_state.json; finished steps are skipped on re-run, and
    interrupted training continues from its own checkpoint. Re-run the SAME command after any failure.
  * Fail early: credentials, GPU, disk and Python packages are checked before any long step.
  * Critical steps stop the run; optional steps (external cohort, XAI, baselines, ...) are recorded as
    failed and the run continues. The reports are always generated, with whatever finished.
  * Everything is logged: one log per step under <run>/logs/, a master log, a step table
    (steps.csv), a resource log (resources.csv), run_meta.json, parameters_master.csv and run_report.pdf.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
PY = sys.executable
REQUIRED_MODULES = {"torch": "torch", "torchvision": "torchvision", "timm": "timm", "cv2": "opencv-python-headless",
                    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy",
                    "matplotlib": "matplotlib", "huggingface_hub": "huggingface_hub", "openpyxl": "openpyxl"}


# ---------------------------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="synthetic data + tiny model on CPU (verifies the pipeline)")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--run-name", default=None, help="default: 'study' (or 'smoke')")
    ap.add_argument("--seeds", default=None, help="comma list; default 1337,2024,7 (smoke: 1337). First = primary")
    ap.add_argument("--limit", type=int, default=15000, help="images in the minority-first cache (1024 px ~ 3.2 MB each)")
    ap.add_argument("--cache-px", type=int, default=1024,
                    help="retinal-field size stored in the cache (1024 ~ 3.2 MB/image, 512 ~ 0.8 MB: "
                         "512 fits ~4x more images on the same disk. Local crops are cut in cache pixels, "
                         "so --crop-px should shrink with it)")
    ap.add_argument("--crop-px", type=int, default=None,
                    help="local window size in cache pixels (default 320 at 1024 px, scaled with --cache-px)")
    ap.add_argument("--drop-path", type=float, default=0.1, help="stochastic depth in the ViT (0 = off)")
    ap.add_argument("--layer-decay", type=float, default=0.8,
                    help="layer-wise LR decay for the backbone (0/1 = off)")
    ap.add_argument("--backbone-lr", type=float, default=5e-5,
                    help="top-block LR when the backbone trains (with layer decay, lower blocks get less)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute ViT activations in the backward pass (~30% slower, much less memory): use on a 24 GB GPU")
    ap.add_argument("--comparison-backbone", default="vit_small_patch16_224",
                    help="backbone of the PROPOSED variants in the comparison step. The default ViT-Small (ImageNet init) is cheap but is NOT "
                         "matched to the plain RETFound ViT-Large baseline; pass vit_large_patch16_224 for a matched, much slower run")
    ap.add_argument("--loss-queue", type=int, default=256,
                    help="sample memory behind the contrastive / DQK / hierarchical-weight losses "
                         "(0 = per-batch statistics, which are degenerate at micro-batch 2-8)")
    ap.add_argument("--no-ensemble", action="store_true", help="skip the seed-ensemble step")
    ap.add_argument("--stage-epochs", default="5,12,14,14,3", help="s1,s2,s3,s4,s5_crt epochs (0 for the last drops cRT)")
    ap.add_argument("--samples-per-epoch", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=None, help="training batch size (default: chosen from GPU memory)")
    ap.add_argument("--accum", type=int, default=None, help="gradient accumulation (default: chosen from GPU memory)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="preprocessing workers")
    ap.add_argument("--train-workers", type=int, default=None, help="dataloader workers (default 4 on Linux+CUDA, else 0)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-cpu", action="store_true", help="permit a non-smoke run without a GPU (impractical for ViT-L)")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--retfound-ckpt", type=Path, default=None)
    ap.add_argument("--comparison-epochs", type=int, default=6)
    ap.add_argument("--skip", default="", help="comma list of step ids to skip")
    ap.add_argument("--only", default="", help="comma list: run only these step ids")
    ap.add_argument("--force", default="", help="comma list of step ids to re-run even if done ('all' for everything)")
    ap.add_argument("--skip-comparison", action="store_true")
    ap.add_argument("--skip-fdr26", action="store_true")
    ap.add_argument("--skip-lesion", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--no-install", action="store_true", help="do not pip install missing packages")
    return ap.parse_args()


# ---------------------------------------------------------------------------------------------
# environment helpers
# ---------------------------------------------------------------------------------------------
def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            return {"cuda": True, "name": pr.name, "mem_gb": round(pr.total_memory / 1e9, 1),
                    "count": torch.cuda.device_count(), "bf16": bool(torch.cuda.is_bf16_supported())}
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        return {"cuda": False, "mps": mps, "name": "mps" if mps else "cpu", "mem_gb": 0.0, "count": 0}
    except Exception as e:                                                  # noqa: BLE001
        return {"cuda": False, "name": f"unknown ({e})", "mem_gb": 0.0, "count": 0}


def auto_batch(g: dict) -> tuple[int, int]:
    """Effective batch stays 16: (per-step batch, accumulation) chosen from GPU memory."""
    m = g.get("mem_gb", 0)
    if m >= 70:
        return 16, 1
    if m >= 38:
        return 8, 2
    if m >= 22:
        return 4, 4
    return 2, 8


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def reachable(host: str, port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------
# step machinery
# ---------------------------------------------------------------------------------------------
@dataclass
class Step:
    id: str
    title: str
    cmd: Callable[[], list] | list | None = None        # subprocess command (lazy so it sees fresh files)
    func: Callable[[], None] | None = None              # or an in-process function
    critical: bool = True
    done: Callable[[], bool] | None = None              # extra proof the step's output exists
    retries: int = 0
    env: dict = field(default_factory=dict)
    enabled: bool = True
    why_skipped: str = ""


class Runner:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.smoke = a.smoke
        self.name = a.run_name or ("smoke" if a.smoke else "study")
        self.data = (a.data_dir or ROOT / ("data_smoke" if a.smoke else "data")).resolve()
        self.run = (ROOT / "outputs" / self.name).resolve()
        self.logs = self.run / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run / "run_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"steps": {}}
        self.gpu = gpu_info()
        seeds = a.seeds or ("1337" if a.smoke else "1337,2024,7")
        self.seeds = [int(x) for x in seeds.split(",") if x.strip()]
        self.first = self.seeds[0]
        bs, accum = auto_batch(self.gpu)
        self.bs = a.bs or (2 if a.smoke else bs)
        self.accum = a.accum or (1 if a.smoke else accum)
        self.amp = (not a.no_amp) and self.gpu.get("cuda", False) and not a.smoke
        self.device = "cpu" if a.smoke else a.device
        self.retfound = (a.retfound_ckpt or self.data / "checkpoints" / "RETFound_MAE" / "pytorch_model.bin")
        self.master = open(self.logs / "run_all.log", "a", encoding="utf-8")
        self.current_step = ""
        self._stop_sampler = threading.Event()
        # paths
        self.raw_eye = self.data / "raw" / ("synthetic" if a.smoke else "eyepacs-hi")
        self.raw_ext = self.data / "raw" / ("synthetic_ext" if a.smoke else "fdr26")
        self.raw_lesion = self.data / "raw" / "lesion"
        self.cache = self.data / "cache"
        self.cache_ext = self.data / "cache_external"
        self.split_file = self.data / "split.json"
        self.lesion_out = self.run / "lesion_pretrain"
        self.final, self.matched = self.run / "final", self.run / "final_matched"
        self.params, self.report = self.run / "parameters", self.run / "report"
        self.cmp = self.run / "comparison"

    # ---- logging -------------------------------------------------------------------------
    def say(self, msg: str = "") -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.master.write(line + "\n")
        self.master.flush()

    def seed_dir(self, s: int) -> Path:
        return self.run / f"seed{s}"

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2))
        with open(self.run / "steps.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "status", "critical", "seconds", "returncode", "started", "log", "note"])
            for k, v in self.state["steps"].items():
                w.writerow([k, v.get("status"), v.get("critical"), v.get("seconds"), v.get("returncode"),
                            v.get("started"), v.get("log"), v.get("note", "")])

    # ---- resource sampler ----------------------------------------------------------------
    def start_sampler(self) -> None:
        path = self.logs / "resources.csv"
        new = not path.exists()

        def loop() -> None:
            try:
                import psutil
            except Exception:                                               # noqa: BLE001
                psutil = None
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "step", "cpu_pct", "ram_used_gb", "gpu_util_pct", "gpu_mem_used_mb", "disk_free_gb"])
                while not self._stop_sampler.wait(60):
                    gu = gm = ""
                    if shutil.which("nvidia-smi"):
                        try:
                            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                            gu, gm = [x.strip() for x in o.stdout.strip().splitlines()[0].split(",")]
                        except Exception:                                   # noqa: BLE001
                            pass
                    w.writerow([datetime.now().isoformat(timespec="seconds"), self.current_step,
                                psutil.cpu_percent() if psutil else "",
                                round(psutil.virtual_memory().used / 1e9, 2) if psutil else "", gu, gm,
                                round(free_gb(self.data), 1)])
                    f.flush()
        threading.Thread(target=loop, daemon=True).start()

    # ---- running one step --------------------------------------------------------------------
    def run_cmd(self, step: Step, log_path: Path) -> int:
        cmd = step.cmd() if callable(step.cmd) else step.cmd
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **step.env}
        with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
            lf.write(f"\n===== {datetime.now().isoformat(timespec='seconds')}  {' '.join(map(str, cmd))}\n")
            lf.flush()
            p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env=env, encoding="utf-8", errors="replace")
            assert p.stdout is not None
            for line in p.stdout:
                lf.write(line)
                sys.stdout.write(f"[{step.id}] {line}")
                sys.stdout.flush()
            return p.wait()

    def execute(self, step: Step) -> bool:
        """Returns True if the run may continue."""
        a = self.a
        force = {x for x in a.force.split(",") if x}
        skip = {x for x in a.skip.split(",") if x}
        only = {x for x in a.only.split(",") if x}
        rec = self.state["steps"].get(step.id, {})
        if only and step.id not in only:
            return True
        if step.id in skip or not step.enabled:
            self.say(f"-- {step.id}: skipped ({step.why_skipped or 'requested'})")
            self.state["steps"][step.id] = {**rec, "status": "skipped", "critical": step.critical,
                                            "note": step.why_skipped or "requested"}
            self.save_state()
            return True
        # summary steps are cheap and must reflect the latest results, so they always re-run
        always = step.id in ("final_stats", "final_ensemble", "matched_stats", "paper_tables", "parameters", "report")
        if (rec.get("status") == "done" and not always and "all" not in force and step.id not in force
                and (step.done is None or step.done())):
            self.say(f"== {step.id}: already done ({rec.get('seconds', '?')}s) - skipping")
            return True
        log = self.logs / f"{step.id}.log"
        self.say(f"== {step.id}: {step.title}" + ("" if step.critical else "   [optional]"))
        self.current_step = step.id
        t0 = time.time()
        started = datetime.now().isoformat(timespec="seconds")
        rc, err = 0, ""
        for attempt in range(step.retries + 1):
            try:
                if step.func is not None:
                    step.func()
                    rc = 0
                else:
                    rc = self.run_cmd(step, log)
            except Exception as e:                                          # noqa: BLE001
                rc, err = 1, f"{type(e).__name__}: {e}"
                with open(log, "a", encoding="utf-8") as lf:
                    lf.write(err + "\n")
            if rc == 0:
                break
            if attempt < step.retries:
                self.say(f"   {step.id}: failed (rc={rc}), retry {attempt + 1}/{step.retries} in 20 s")
                time.sleep(20)
        dt = round(time.time() - t0, 1)
        ok = rc == 0 and (step.done is None or step.done())
        if rc == 0 and not ok:
            err = err or "step exited 0 but its expected output is missing"
        self.state["steps"][step.id] = {
            "status": "done" if ok else ("failed" if step.critical else "failed_optional"),
            "critical": step.critical, "seconds": dt, "returncode": rc, "started": started,
            "log": str(log.relative_to(self.run)), "note": err}
        self.save_state()
        self.say(f"   {step.id}: {'OK' if ok else 'FAILED'} in {dt}s" + (f" ({err})" if err and not ok else ""))
        if not ok and step.critical:
            self.say(f"!! critical step '{step.id}' failed - see {log}. Fix the cause and re-run the SAME command: "
                     f"finished steps are skipped and training resumes from its checkpoint.")
            return False
        return True


# ---------------------------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------------------------
def preflight(r: Runner) -> bool:
    a = r.a
    problems: list[str] = []
    warns: list[str] = []
    r.say(f"run '{r.name}'  mode={'SMOKE (synthetic)' if r.smoke else 'FULL'}  seeds={r.seeds}  data={r.data}")
    if sys.version_info < (3, 10):
        problems.append(f"Python >= 3.10 required, found {platform.python_version()}")

    # packages
    def missing_mods() -> list[str]:
        out = []
        for mod, pipname in REQUIRED_MODULES.items():
            try:
                __import__(mod)
            except Exception:                                               # noqa: BLE001
                out.append(pipname)
        return out
    miss = missing_mods()
    if miss and not a.no_install:
        r.say(f"installing missing packages: {miss}")
        rc = subprocess.call([PY, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
        if rc != 0:
            problems.append("pip install -r requirements.txt failed")
        miss = missing_mods()
    if miss:
        problems.append(f"missing Python packages: {miss} (run: pip install -r requirements.txt)")
    r.gpu = gpu_info()
    r.say(f"device: {r.gpu}")
    if not r.smoke and not r.gpu.get("cuda") and not a.allow_cpu:
        problems.append("no CUDA GPU detected. A ViT-Large run on CPU is impractical; use a GPU instance, "
                        "or pass --allow-cpu to proceed anyway, or --smoke for the synthetic test.")
    if r.gpu.get("cuda") and r.gpu["mem_gb"] < 20 and not r.smoke:
        warns.append(f"GPU has {r.gpu['mem_gb']} GB; batch {r.bs} x accum {r.accum} is used. "
                     "OOM is possible at 512 px; lower --bs or --global-size if it happens")

    # disk
    need = 0.5 if r.smoke else (8 + 8 + a.limit * 3.15 * (a.cache_px / 1024) ** 2 / 1000 + 5 + 4 + 3 + 6)
    have = free_gb(r.data)
    r.say(f"disk: {have:.0f} GB free in {r.data}, estimated need {need:.0f} GB")
    if have < need:
        problems.append(f"not enough disk: {have:.0f} GB free, ~{need:.0f} GB needed (lower --limit or attach a bigger volume)")

    # credentials (only if the thing they fetch is not already present)
    if not r.smoke:
        have_kaggle = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or \
            (Path.home() / ".kaggle" / "kaggle.json").exists()
        have_eye = (r.raw_eye.exists() and any(r.raw_eye.rglob("trainLabels.csv")))
        if not have_eye and not have_kaggle:
            problems.append("EyePACS is not downloaded and Kaggle credentials are missing: set KAGGLE_USERNAME and "
                            "KAGGLE_KEY (Kaggle -> Account -> Create New API Token)")
        have_rf = r.retfound.exists() and r.retfound.stat().st_size > 5e8
        if not have_rf and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            problems.append("RETFound weights are missing and HF_TOKEN is not set: request access to "
                            "YukunZhou/RETFound_mae_natureCFP on HuggingFace, create a read token, export HF_TOKEN")
        hosts = ["www.kaggle.com", "zenodo.org", "huggingface.co", "pypi.org"]
        down = [h for h in hosts if not reachable(h)]
        if down:
            warns.append(f"cannot reach {down}: downloads from those hosts will fail")

    for w in warns:
        r.say(f"WARNING: {w}")
    for p_ in problems:
        r.say(f"PROBLEM: {p_}")

    meta = {"started": datetime.now().isoformat(timespec="seconds"), "args": {k: str(v) for k, v in vars(a).items()},
            "python": platform.python_version(), "platform": platform.platform(), "gpu": r.gpu,
            "batch_size": r.bs, "grad_accum": r.accum, "amp": r.amp, "seeds": r.seeds, "data_dir": str(r.data),
            "problems": problems, "warnings": warns}
    try:
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                                            timeout=10).stdout.strip()
    except Exception:                                                       # noqa: BLE001
        pass
    for mod in ("torch", "torchvision", "timm", "numpy", "pandas", "sklearn", "cv2"):
        try:
            meta[f"{mod}_version"] = __import__(mod).__version__
        except Exception:                                                   # noqa: BLE001
            pass
    (r.run / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if problems:
        r.say(f"preflight FAILED ({len(problems)} problem(s)) - nothing was started.")
        return False
    r.say("preflight OK")
    return True


# ---------------------------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------------------------
def build_plan(r: Runner) -> list[Step]:
    a = r.a
    S = f"{ROOT / 'scripts'}"
    sm = r.smoke
    nc = 4 if sm else 8                      # local crops
    glob, cin = (64, 32) if sm else (512, 288)
    dev = ["--device", r.device]
    steps: list[Step] = []

    def add(*args, **kw) -> Step:
        s = Step(*args, **kw)
        steps.append(s)
        return s

    # -- tests and data ------------------------------------------------------------------------------
    add("selftest", "architecture self-test (all blocks on synthetic tensors)", cmd=[PY, f"{S}/08_selftest.py"],
        enabled=not a.skip_selftest, why_skipped="--skip-selftest")
    if sm:
        add("data_synth", "generate synthetic fundus data (EyePACS layout + DDR-like lesion masks)",
            cmd=[PY, f"{S}/00_make_synthetic_data.py", "--out", r.raw_eye, "--lesion-out", r.raw_lesion,
                 "--n-patients", "40", "--size", "512"], done=lambda: (r.raw_eye / "trainLabels.csv").exists())
        add("data_synth_ext", "generate a second synthetic cohort (stands in for the external cohort)",
            cmd=[PY, f"{S}/00_make_synthetic_data.py", "--out", r.raw_ext, "--n-patients", "20", "--size", "512",
                 "--seed", "99"], done=lambda: (r.raw_ext / "trainLabels.csv").exists(), critical=False)
    else:
        add("download_eyepacs", "download EyePACS (1024 px mirror, ~7.8 GB)",
            cmd=[PY, f"{S}/01_download_data.py", "--dataset", "eyepacs-hi", "--out", r.data / "raw"], retries=2,
            done=lambda: any(r.raw_eye.rglob("trainLabels.csv")) if r.raw_eye.exists() else False)
        add("download_retfound", "download RETFound CFP weights (gated)",
            cmd=[PY, f"{S}/01b_download_retfound.py", "--out", r.retfound], retries=1,
            done=lambda: r.retfound.exists() and r.retfound.stat().st_size > 5e8)
        add("download_lesions", "download DDR lesion annotations", cmd=[PY, f"{S}/11_download_lesions.py", "--out", r.raw_lesion],
            critical=False, retries=2, enabled=not a.skip_lesion, why_skipped="--skip-lesion")
        add("download_fdr26", "download the FDR26 external cohort (Zenodo)",
            cmd=[PY, f"{S}/01_download_data.py", "--dataset", "fdr26", "--out", r.data / "raw"],
            critical=False, retries=2, enabled=not a.skip_fdr26, why_skipped="--skip-fdr26",
            done=lambda: r.raw_ext.exists() and any(r.raw_ext.rglob("*.jpg")))

    # -- preprocessing -----------------------------------------------------------------------------------
    cache_px = 512 if sm else a.cache_px
    crop_px = a.crop_px or int(round(320 * cache_px / 1024 / 8) * 8)
    pre = [PY, f"{S}/02_preprocess.py", "--root", r.raw_eye, "--out", r.cache, "--image-size", cache_px,
           "--n-crops", nc, "--workers", 1 if sm else a.workers, "--minority-first", "--split-file", r.split_file,
           "--limit", 64 if sm else a.limit, "--seed", r.first]
    if sm:
        pre += ["--calibration-sample", "20"]
    else:
        pre += ["--crop-size", crop_px]
    add("preprocess", "A1 quality gate + A2 field extraction + lesion priors + crop centres -> cache", cmd=pre,
        done=lambda: (r.cache / "meta.csv").exists())
    ext_pre = ([PY, f"{S}/02_preprocess.py", "--root", r.raw_ext, "--out", r.cache_ext, "--image-size",
                cache_px, "--n-crops", nc] + ([] if sm else ["--crop-size", crop_px]) + [ "--workers", 1 if sm else a.workers, "--seed", r.first,
                "--quality-reference", r.cache / "quality_reference.json"]
               + (["--kaggle-test", "--labels-csv", r.raw_ext / "trainLabels.csv"] if sm else ["--fdr26"]))
    add("preprocess_external", "preprocess the external cohort", cmd=ext_pre, critical=False,
        enabled=not a.skip_fdr26, why_skipped="--skip-fdr26", done=lambda: (r.cache_ext / "meta.csv").exists())

    # -- lesion pretraining ------------------------------------------------------------------------------------
    les = [PY, f"{S}/10_lesion_pretrain.py", "--root", r.raw_lesion, "--out", r.lesion_out, "--image-size", cin,
           "--batch-size", 4 if sm else 6, "--epochs", 1 if sm else 10, "--crops-per-image", 4 if sm else 8, *dev]
    add("lesion_pretrain", "pretrain the A3 lesion experts on real expert masks", cmd=les, critical=False,
        enabled=not a.skip_lesion, why_skipped="--skip-lesion", done=lambda: (r.lesion_out / "lesion_encoder.pt").exists())

    # -- training: one run per seed ----------------------------------------------------------------------------------
    for s in r.seeds:
        sd = r.seed_dir(s)

        def train_cmd(s=s, sd=sd):
            c = [PY, f"{S}/03_train.py", "--cache", r.cache, "--out", r.run, "--tag", f"seed{s}", "--seed", s,
                 "--batch-size", r.bs, "--grad-accum", r.accum, "--global-size", glob, "--crop-input", cin,
                 "--n-crops", nc, *dev, "--workers", 0 if sm else (a.train_workers if a.train_workers is not None else
                                                                    (4 if (r.gpu.get("cuda") and os.name != "nt") else 0))]
            if sm:
                c += ["--backbone", "vit_tiny_patch16_224", "--retfound-ckpt", "none", "--allow-no-retfound",
                      "--stage-epochs", "1,1,1,1,1", "--samples-per-epoch", 8, "--max-val", 8]
            else:
                c += ["--retfound-ckpt", r.retfound, "--stage-epochs", a.stage_epochs,
                      "--samples-per-epoch", a.samples_per_epoch, "--max-val", 5000]
                if r.amp:
                    c += ["--amp"]
            if (r.lesion_out / "lesion_encoder.pt").exists():
                c += ["--lesion-encoder", r.lesion_out / "lesion_encoder.pt"]
            # regularisation / optimisation for fine-tuning the large ViT (see TrainCfg.layer_decay)
            c += ["--drop-path", a.drop_path, "--layer-decay", a.layer_decay, "--backbone-lr", a.backbone_lr,
                  "--loss-queue", a.loss_queue]
            if not sm:
                c += ["--crop-size", crop_px]          # window size must follow the cache resolution
            if a.grad_checkpoint:
                c += ["--grad-checkpoint"]
            return c
        add(f"train_seed{s}", f"staged training, seed {s} (frozen -> LoRA -> unfreeze -> full -> decoupled head)",
            cmd=train_cmd, done=lambda sd=sd: (sd / "best.pt").exists())

        tta = ["--tta"] if not sm else []
        add(f"eval_val_seed{s}", f"validation predictions, seed {s}",
            cmd=[PY, f"{S}/04_evaluate.py", "--ckpt", sd / "best.pt", "--cache", r.cache, "--split", "val",
                 "--predictions-only", *tta, "--out", sd / "evaluation_val.json", *dev],
            done=lambda sd=sd: any(sd.glob("predictions_val*.npz")))
        add(f"eval_test_seed{s}", f"full test evaluation, seed {s} (38 parameters, XAI, robustness, deployment)",
            cmd=[PY, f"{S}/04_evaluate.py", "--ckpt", sd / "best.pt", "--cache", r.cache, "--split", "test", *tta,
                 "--xai-samples", 6 if sm else 120, "--corruption-samples", 8 if sm else 400,
                 "--mc-samples", 2 if sm else 8, "--out", sd / "evaluation.json", *dev],
            done=lambda sd=sd: (sd / "evaluation.json").exists())
        add(f"eval_external_seed{s}", f"external cohort predictions, seed {s}",
            cmd=[PY, f"{S}/04_evaluate.py", "--ckpt", sd / "best.pt", "--cache", r.cache_ext, "--split", "test",
                 "--predictions-only", *tta, "--pred-tag", "external", "--out", sd / "evaluation_external.json", *dev],
            critical=False, enabled=not a.skip_fdr26, why_skipped="--skip-fdr26",
            done=lambda sd=sd: any(sd.glob("predictions_*external*.npz")))

    sd1 = r.seed_dir(r.first)
    add("screening", "screening triage: validation-selected thresholds on test",
        cmd=[PY, f"{S}/09_screening.py", "--ckpt", sd1 / "best.pt", "--cache", r.cache,
             *(["--external-cache", r.cache_ext] if (not a.skip_fdr26) else []), *dev], critical=False)
    add("xai_expert_masks", "XAI faithfulness against held-out expert lesion masks",
        cmd=[PY, f"{S}/21_xai_evaluation.py", "--ckpt", sd1 / "best.pt", "--cache", r.cache, "--ref-masks", "idrid_ddr",
             "--lesion-root", r.raw_lesion, "--n", 8 if sm else 120, "--device", "cpu" if sm else "cuda"
             if r.gpu.get("cuda") else "cpu", "--out", sd1 / "xai_evaluation_expert_masks.json"],
        critical=False, enabled=not a.skip_lesion, why_skipped="--skip-lesion")
    add("train_predictions", "training-split predictions (input to the label-noise pass)",
        cmd=[PY, f"{S}/04_evaluate.py", "--ckpt", sd1 / "best.pt", "--cache", r.cache, "--split", "train",
             "--predictions-only", "--out", sd1 / "evaluation_train.json", *dev], critical=False,
        done=lambda: (sd1 / "predictions_train.npz").exists())
    add("label_noise", "flag probable label errors (review queue + sampling weights)",
        cmd=[PY, f"{S}/27_flag_label_noise.py", "--pred", sd1 / "predictions_train.npz", "--out", r.run / "noise_weights.csv"],
        critical=False, done=lambda: (r.run / "noise_weights.csv").exists())

    cmp_cmd = [PY, f"{S}/07_compare_models.py", "--cache", r.cache, "--epochs", 1 if sm else a.comparison_epochs,
               "--max-train", 0, "--batch-size", 32, "--image-size", 224, "--seed", r.first, "--retfound-ckpt",
               r.retfound, "--backbone", a.comparison_backbone, "--out", r.cmp, *dev]
    add("comparison", "matched-budget baselines and ablations (incl. plain RETFound)", cmd=cmp_cmd, critical=False,
        enabled=not (a.skip_comparison or sm),
        why_skipped="--skip-comparison" if a.skip_comparison else "smoke mode: baselines need pretrained weights",
        # the headline entry must exist: 07 records per-model failures and still exits 0
        done=lambda: (r.cmp / "comparison.json").exists() and (r.cmp / "predictions_complete.npz").exists())

    # -- statistics, parameters, report ------------------------------------------------------------------------------------
    # -- seed ensemble: average the members' probabilities, then decode / calibrate / screen on the
    #    ensemble's own validation scores (never on one member's)
    ens = r.run / "ensemble"
    use_ens = len(r.seeds) > 1 and not a.no_ensemble

    def member_files(kind: str) -> list:
        pat = {"val": "predictions_val*.npz", "test": "predictions_test*.npz",
               "external": "predictions_*external*.npz"}[kind]
        out = []
        for s_ in r.seeds:
            c = sorted(p for p in r.seed_dir(s_).glob(pat) if ("external" in p.name) == (kind == "external"))
            if c:
                out.append(c[0])
        return out

    def ens_cmd(kind: str):
        def f() -> list:
            files = member_files(kind)
            if len(files) < 2:
                raise FileNotFoundError(f"only {len(files)} seed(s) have {kind} predictions; need >= 2 to ensemble")
            return [PY, f"{S}/31_ensemble.py", "--inputs", *files, "--out", ens / f"predictions_{kind}.npz"]
        return f
    if use_ens:
        for kind in ("val", "test", "external"):
            add(f"ensemble_{kind}", f"average {kind} probabilities over seeds", cmd=ens_cmd(kind), critical=False,
                enabled=(kind != "external" or not a.skip_fdr26), why_skipped="--skip-fdr26",
                done=lambda kind=kind: (ens / f"predictions_{kind}.npz").exists())

    def final_ens_cmd() -> list:
        c = [PY, f"{S}/26_final_evaluation.py", "--val", ens / "predictions_val.npz",
             "--test", ens / "predictions_test.npz", "--out", r.run / "final_ensemble"]
        if (ens / "predictions_external.npz").exists():
            c += ["--external", f"external={ens / 'predictions_external.npz'}"]
        return c
    add("final_ensemble", "final statistics on the seed ensemble (decoders fitted on the ensemble's validation)",
        cmd=final_ens_cmd, critical=False, enabled=use_ens,
        why_skipped="single seed or --no-ensemble",
        done=lambda: (r.run / "final_ensemble" / "final_evaluation.json").exists())

    def first(paths, what: str):
        for p_ in paths:
            return p_
        raise FileNotFoundError(f"no {what} predictions in {sd1} - the evaluation step that writes them "
                                f"has not completed (see its log under {r.logs})")

    def final_cmd() -> list:
        c = [PY, f"{S}/26_final_evaluation.py", "--val", first(sd1.glob("predictions_val*.npz"), "validation"),
             "--test", first((p for p in sd1.glob("predictions_test*.npz") if "external" not in p.name), "test"),
             "--out", r.final]
        ext = list(sd1.glob("predictions_*external*.npz"))
        if ext:
            c += ["--external", f"external={ext[0]}"]
        others = [next((p for p in r.seed_dir(s).glob("predictions_test*.npz") if "external" not in p.name), None)
                  for s in r.seeds[1:]]
        others = [o for o in others if o]
        if others:
            c += ["--seed-test", *others]
        return c
    add("final_stats", "final statistics: patient-level CIs, validation-fitted decoders, screening, seed stability",
        cmd=final_cmd, critical=False, done=lambda: (r.final / "final_evaluation.json").exists(), env={})

    def matched_cmd() -> list:
        c = [PY, f"{S}/26_final_evaluation.py", "--test", r.cmp / "predictions_complete.npz", "--out", r.matched]
        for f in sorted(r.cmp.glob("predictions_*.npz")):
            n = f.stem.replace("predictions_", "")
            if n != "complete":
                c += ["--baseline", f"{n}={f}"]
        return c
    add("matched_stats", "matched-budget paired comparison with Holm correction", cmd=matched_cmd, critical=False,
        enabled=not (a.skip_comparison or sm), why_skipped="no comparison run",
        done=lambda: (r.matched / "final_evaluation.json").exists())
    add("paper_tables", "per-stage / best-value tables and the score-vs-time figure",
        cmd=[PY, f"{S}/25_paper_tables.py", "--run", sd1], critical=False)
    add("parameters", "master parameter log (every parameter, with provenance)",
        cmd=[PY, f"{S}/29_collect_parameters.py", "--run", sd1, "--final", r.final, "--matched", r.matched,
             "--cache", r.cache, "--lesion-out", r.lesion_out, "--out", r.params], critical=False,
        done=lambda: (r.params / "parameters_master.csv").exists())
    add("report", "PDF report + CSV tables",
        cmd=[PY, f"{S}/30_run_report.py", "--run", sd1, "--study", r.run, "--cache", r.cache,
             "--params", r.params / "parameters_master.csv", "--out", r.report], critical=False,
        done=lambda: (r.report / "run_report.pdf").exists())
    return steps


# ---------------------------------------------------------------------------------------------
# results index
# ---------------------------------------------------------------------------------------------
def write_index(r: Runner, ok: bool) -> None:
    lines = [f"# Run '{r.name}' - {'COMPLETE' if ok else 'INCOMPLETE'}", "", f"written {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             "## Steps", "", "| step | status | seconds | note |", "|---|---|---|---|"]
    for k, v in r.state["steps"].items():
        lines.append(f"| {k} | {v.get('status')} | {v.get('seconds', '')} | {str(v.get('note', ''))[:80]} |")
    lines += ["", "## Key outputs", ""]
    for rel in ("report/run_report.pdf", "parameters/parameters_master.csv", "final/final_evaluation.md",
                "final_ensemble/final_evaluation.md",
                "final_matched/final_evaluation.md", "steps.csv", "run_meta.json", "logs/run_all.log", "logs/resources.csv",
                "noise_weights.csv"):
        p = r.run / rel
        lines.append(f"- `{rel}` {'(' + str(round(p.stat().st_size / 1024)) + ' KB)' if p.exists() else '- not produced'}")
    for s in r.seeds:
        sd = r.seed_dir(s)
        lines.append(f"- `seed{s}/`: " + (", ".join(sorted(x.name for x in sd.glob('*') if x.is_file()))[:300] if sd.exists() else "not produced"))
    (r.run / "RESULTS_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    a = parse_args()
    r = Runner(a)
    r.say("=" * 78)
    if not preflight(r):
        return 2
    if a.preflight_only:
        return 0
    r.start_sampler()
    plan = build_plan(r)
    ok = True
    try:
        for st in plan:
            if not r.execute(st):
                ok = False
                break
    except KeyboardInterrupt:
        r.say("interrupted - re-run the same command to resume")
        ok = False
    finally:
        # reports are produced even after a failure, from whatever finished
        if not ok:
            for sid in ("final_stats", "paper_tables", "parameters", "report"):
                st = next((s for s in plan if s.id == sid), None)
                if st is not None and r.state["steps"].get(sid, {}).get("status") != "done":
                    try:
                        r.execute(st)
                    except Exception as e:                                  # noqa: BLE001
                        r.say(f"could not build {sid}: {e}")
        r._stop_sampler.set()
        write_index(r, ok)
        r.say(f"index: {r.run / 'RESULTS_INDEX.md'}")
        r.master.close()
    failed = [k for k, v in r.state["steps"].items() if v.get("status") == "failed"]
    return 0 if ok and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
