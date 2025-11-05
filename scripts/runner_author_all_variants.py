#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import itertools
import subprocess
import datetime
import traceback
from pathlib import Path
from multiprocessing import Pool

BASE_DIR = "/opt/gpudata/summereunann/echonext_suite"
SCRIPT = f"{BASE_DIR}/scripts/protoecgnet_original/main.py"

RESULTS_ROOT = (
    f"{BASE_DIR}/results/ultimate_comparison/protoecgnet/protoecgnet_all"
)

# 👇 THIS was missing
PROCESSED_DIR = f"{BASE_DIR}/processed"

GPUS = [0, 1, 2, 3]

backbones = [
    "resnet1d18",
    "resnet1d34",
    "resnet1d50",
    "simple_cnn",
    "deep_cnn",
    "cnn_transformer",
]

modes = ["binary"]
protos = [5, 10]

seed = 42
emb = 256
lrenc = 3e-4
lrhead = 3e-4
clst = 0.8
sep = 0.08
dimension = "1D"
training_stage = "joint"
sampling_rate = 500
label_set = "all"

FORCE_RERUN = os.environ.get("FORCE_RERUN", "0") == "1"


def exp_name(mode, backbone, proto):
    return (
        f"mode{mode}_backbone{backbone}_protos{proto}"
        f"_emb{emb}_lrenc{lrenc:.0e}_lrhead{lrhead:.0e}"
        f"_clust{clst}_sep{sep}_seed{seed}"
    )


def run_job(args):
    mode, backbone, proto, gpu_id, log_path = args
    exp = exp_name(mode, backbone, proto)

    run_dir = Path(RESULTS_ROOT) / exp
    run_dir.mkdir(parents=True, exist_ok=True)

    # main.py always nests another folder named exp
    nested_dir = run_dir / exp
    metrics_path = nested_dir / f"{exp}_metrics.json"

    if metrics_path.exists() and not FORCE_RERUN:
        msg = f"[SKIP] {exp} – metrics already exists at {metrics_path}"
        print(msg)
        with open(log_path, "a") as lf:
            lf.write(msg + "\n")
        return

    cmd = [
        "python",
        SCRIPT,
        "--job_name",
        exp,
        "--epochs",
        "30",
        "--batch_size",
        "64",
        "--lr",
        str(lrenc),
        "--lam_clst",
        str(clst),
        "--lam_sep",
        str(sep),
        "--proto_dim",
        str(emb),
        "--proto_time_len",
        "3",
        "--single_class_prototype_per_class",
        str(proto),
        "--dimension",
        dimension,
        "--training_stage",
        training_stage,
        "--backbone",
        backbone,
        "--sampling_rate",
        str(sampling_rate),
        "--label_set",
        label_set,
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--checkpoint_dir",
        str(run_dir),
        "--log_dir",
        str(run_dir),
        "--test_dir",
        str(run_dir),
        "--processed_dir",
        PROCESSED_DIR,          # 👈 this fixes the “could not build dataloaders”
        "--mode",
        mode,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"\n🚀 Launching {exp} on GPU {gpu_id}")
    with open(log_path, "a") as lf:
        lf.write(f"\n[{datetime.datetime.now()}] === {exp} on GPU {gpu_id} ===\n")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
            for line in proc.stdout:
                line = line.rstrip()
                print(f"[GPU{gpu_id}] {line}")
                lf.write(line + "\n")
            proc.wait()
            if proc.returncode != 0:
                lf.write(f"[ERROR] {exp} exited with code {proc.returncode}\n")
        except Exception as e:
            lf.write(f"[FATAL] {exp} crashed: {e}\n{traceback.format_exc()}\n")


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(RESULTS_ROOT) / f"runs_parallel_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = list(itertools.product(modes, backbones, protos))
    print(f"Launching {len(jobs)} jobs across {len(GPUS)} GPUs → {log_path}\n")

    tasks = [
        (mode, backbone, proto, GPUS[i % len(GPUS)], log_path)
        for i, (mode, backbone, proto) in enumerate(jobs)
    ]

    from multiprocessing import Pool
    with Pool(processes=len(GPUS)) as pool:
        pool.map(run_job, tasks)

    print(f"\n✅ All runs complete. Logs saved at: {log_path}")


if __name__ == "__main__":
    main()