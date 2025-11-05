#!/usr/bin/env python3
# main.py — robust to proto model kwargs, matches training_functions.py,
# uses processed_dir, and (now) actually finds the test CSV trainer saved.

if __name__ == "__main__":
    import argparse
    import os
    import sys as _sys
    from pathlib import Path as _Path

    import numpy as np
    import torch

    # big training file showed
    from training_functions import (
        train_model,
        test_model,
        seed_everything,
    )

    from ecg_utils import get_dataloaders, load_label_mappings
    from proto_models1D import ProtoECGNet1D
    from proto_models2D import ProtoECGNet2D
    from fusion import (
        FusionProtoClassifier,
        load_fusion_label_mappings,
        get_fusion_dataloaders,
    )

    # --- Import ResNet backbones so names exist in the namespace ---
    from backbones import (
        resnet1d18,
        resnet1d34,
        resnet1d50,
        resnet1d101,
        resnet1d152,
        resnet18,
        resnet34,
        resnet50,
        resnet101,
        resnet152,
    )

    # --- Try to import extra CNN backbones (not required) ---
    try:
        from custom_backbones import (
            simple_cnn,
            deep_cnn,
            wide_cnn,
            cnn_gru,
            cnn_lstm,
            cnn_transformer,
            inception_cnn,
        )
        print("[INFO] Custom CNN backbones loaded successfully.")
    except ImportError as e:
        print(f"[INFO] Skipping custom_backbones.py ({e}). Only ResNet backbones will be used.")

    # speed tweak
    torch.set_float32_matmul_precision("high")

    # === make /opt/gpudata/summereunann/echonext_suite/scripts/metrics.py importable ===
    _here = _Path(__file__).resolve()
    _project_guess_1 = _here.parents[2]
    _project_guess_2 = _here.parents[1]
    if (_project_guess_1 / "scripts").exists():
        _proj_root = _project_guess_1
    else:
        _proj_root = _project_guess_2
    _scripts_dir = _proj_root / "scripts"
    if str(_scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(_scripts_dir))
    try:
        # this is separate metrics.py (the one pasted)
        from metrics import compute_metrics_dict as _compute_metrics_dict
    except Exception:
        _compute_metrics_dict = None

    # ---------------------------------------------------------------
    # metrics helper — FIXED to:
    #   1) look in <test_dir>/<job_name>/... (what trainer does)
    #   2) understand Label_*/Prob_* columns
    # ---------------------------------------------------------------
    def _run_metrics_from_csv(args):
        if _compute_metrics_dict is None:
            print("[metrics] metrics.py not available, skipping.")
            return

        from pathlib import Path as _P2
        import pandas as _pd
        import json as _json

        test_dir = _P2(args.test_dir)
        job = args.job_name

        # trainer wrote: <test_dir>/<job>/<job>_test_results_v0.csv
        job_dir = test_dir / job

        candidates = []

        # a) the exact path trainer wrote in logs
        candidates.append(job_dir / f"{job}_test_results_v0.csv")
        # b) any versioned file (sometimes PL increments version)
        if job_dir.exists():
            candidates.extend(sorted(job_dir.glob(f"{job}_test_results_v*.csv")))
        # c) old flat layout, just in case
        candidates.append(test_dir / f"{job}_test_results_v0.csv")

        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break

        if csv_path is None:
            print(f"[metrics] CSV not found in {job_dir} or {test_dir}, skipping.")
            return

        print(f"[metrics] reading {csv_path}")
        df = _pd.read_csv(csv_path)
        cols = list(df.columns)

        mode = getattr(args, "mode", "multilabel").lower()

        y_true = y_prob = None
        valid_mask = None

        # -------- CASE 1: trainer-style multilabel: Label_xxx / Prob_xxx --------
        label_like = sorted([c for c in cols if c.startswith("Label_")])
        prob_like = sorted([c for c in cols if c.startswith("Prob_")])
        if label_like and prob_like and len(label_like) == len(prob_like):
            # build arrays in a consistent order
            Y_true = df[label_like].to_numpy().astype(float)
            Y_prob = df[prob_like].to_numpy().astype(float)
            y_true, y_prob = Y_true, Y_prob
            # since this is the format  trainer uses, we can just run multilabel
            metrics_dict = _compute_metrics_dict("multilabel", y_true, y_prob, valid_mask=valid_mask)
            out_path = job_dir / f"{job}_metrics.json"
            out_path.write_text(_json.dumps(metrics_dict, indent=2))
            print(f"[metrics] saved metrics to {out_path}")
            return

        # -------- CASE 2: earlier runner-style CSV names --------
        if mode == "binary":
            truth_candidates = ["y_true", "label", "target", "gt"]
            prob_candidates = ["y_prob", "prob", "pred_prob", "pred", "score", "logit"]
            for c in truth_candidates:
                if c in df:
                    y_true = df[c].to_numpy().astype(float)
                    break
            for c in prob_candidates:
                if c in df:
                    y_prob = df[c].to_numpy().astype(float)
                    break
            if y_true is None or y_prob is None:
                # fall back to multilabel inference
                mode = "multilabel"

        if mode != "binary" and (y_true is None or y_prob is None):
            true_like = sorted([c for c in cols if c.startswith("y_true_") or c.endswith("_true")])
            prob_like = sorted([c for c in cols if c.startswith("y_prob_") or c.endswith("_prob")])
            if true_like and prob_like and len(true_like) == len(prob_like):
                y_true = df[true_like].to_numpy().astype(float)
                y_prob = df[prob_like].to_numpy().astype(float)
            else:
                print("[metrics] could not infer columns, skipping.")
                return

        metrics_dict = _compute_metrics_dict(mode, y_true, y_prob, valid_mask=valid_mask)
        # for runner-style, write next to test_dir
        out_path = test_dir / f"{job}_metrics.json"
        out_path.write_text(_json.dumps(metrics_dict, indent=2))
        print(f"[metrics] saved metrics to {out_path}")

    # ================================================================
    # argument parser
    # ================================================================
    parser = argparse.ArgumentParser(description="Train and/or test ProtoECGNet models.")

    # general
    parser.add_argument("--job_name", type=str, default="protoecgnet_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dimension", type=str, default="1D", choices=["1D", "2D"])
    parser.add_argument("--mode", type=str, default="multilabel", choices=["binary", "multilabel", "multitask"])
    parser.add_argument("--label_set", type=str, default="all")
    parser.add_argument("--custom_groups", action="store_true", default=False)

    # data / paths
    parser.add_argument("--train_dir", type=str, default="/opt/gpudata/summereunann/echonext_suite/data/train")
    parser.add_argument("--val_dir", type=str, default="/opt/gpudata/summereunann/echonext_suite/data/val")
    parser.add_argument("--test_dir", type=str, default="/opt/gpudata/summereunann/echonext_suite/data/test")
    parser.add_argument("--checkpoint_dir", type=str, default="/opt/gpudata/summereunann/echonext_suite/checkpoints")
    parser.add_argument("--log_dir", type=str, default="/opt/gpudata/summereunann/echonext_suite/logs")
    parser.add_argument("--processed_dir", type=str, default=None)

    # training params
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--backbone", type=str, default="simple_cnn")
    parser.add_argument("--sampling_rate", type=int, default=500)

    # prototype-related
    parser.add_argument("--proto_dim", type=int, default=256)
    parser.add_argument("--proto_time_len", type=int, default=3)
    parser.add_argument("--single_class_prototype_per_class", type=int, default=10)

    # loss weights
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--lam_clst", type=float, default=0.8)
    parser.add_argument("--lam_sep", type=float, default=0.08)
    parser.add_argument("--lam_spars", type=float, default=0.0)
    parser.add_argument("--lam_div", type=float, default=0.0)
    parser.add_argument("--lam_cnrst", type=float, default=0.0)

    # lightning / trainer knobs training_functions.py expects
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume_checkpoint", action="store_true", help="resume last ckpt if present")
    parser.add_argument("--save_weights", action="store_true", help="save model weights after projection")
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="none",
        choices=["none", "ReduceLROnPlateau", "CosineAnnealingLR", "CyclicLR"],
    )

    # training stage
    parser.add_argument(
        "--training_stage",
        type=str,
        default="joint",
        choices=["projection", "fusion", "joint", "prototypes", "classifier", "feature_extractor"],
    )

    # testing only
    parser.add_argument("--test_model", action="store_true", help="If set, only run test_model().")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # patch a few lightning-ish attrs so we don't crash later
    # ------------------------------------------------------------
    if not hasattr(args, "log_every_n_steps"):
        args.log_every_n_steps = 50
    if not hasattr(args, "check_val_every_n_epoch"):
        args.check_val_every_n_epoch = 1
    if not hasattr(args, "num_sanity_val_steps"):
        args.num_sanity_val_steps = 0

    # ================================================================
    # main logic
    # ================================================================
    seed_everything(args.seed)

    # make training_functions.get_class_names() see the same processed dir
    if args.processed_dir:
        os.environ["ECHONEXT_PROCESSED_DIR"] = args.processed_dir

    _processed_dir = args.processed_dir  # may be None → ecg_utils will use default

    # ----------------------------------------------------------------
    # helper to build class weights from label_map (what train_model wants)
    # ----------------------------------------------------------------
    def _make_class_weights(label_map):
        y = np.asarray(label_map["train_labels"])
        if y.ndim == 1:
            y = y[:, None]
        pos = y.sum(axis=0)
        n = y.shape[0]
        weights = (n - pos) / (pos + 1e-6)
        return torch.from_numpy(weights.astype(np.float32))

    # ----------------------------------------------------------------
    # robust model builder
    # ----------------------------------------------------------------
    def _build_proto_model(args, num_classes: int):
        if args.dimension.upper() == "1D":
            cls = ProtoECGNet1D
        else:
            cls = ProtoECGNet2D

        base = dict(
            num_classes=num_classes,
            proto_dim=args.proto_dim,
            single_class_prototype_per_class=args.single_class_prototype_per_class,
        )

        candidate_kwargs = [
            dict(base, backbone=args.backbone, mode=args.mode, proto_time_len=args.proto_time_len),
            dict(base, backbone=args.backbone, mode=args.mode),
            dict(base, backbone=args.backbone),
            dict(base, mode=args.mode),
            dict(base),
        ]

        last_err = None
        for kw in candidate_kwargs:
            try:
                return cls(**kw)
            except TypeError as e:
                last_err = e
                continue
        raise last_err

    # ----- test-only path ----------------------------------------------------
    if args.test_model:
        label_map = load_label_mappings(
            processed_dir=_processed_dir,
            mode=args.mode,
            label_set=args.label_set,
            custom_groups=args.custom_groups,
        )
        train_loader, val_loader, test_loader = get_dataloaders(args, label_map)

        model = _build_proto_model(args, num_classes=len(label_map["all"]))
        model.to(args.device)

        test_model(model, test_loader, args=args)
        _run_metrics_from_csv(args)
        raise SystemExit(0)

    # ----- fusion stage ------------------------------------------------------
    if args.training_stage == "fusion":
        fusion_label_map = load_fusion_label_mappings(args.label_set)
        train_loader, val_loader, test_loader = get_fusion_dataloaders(args, fusion_label_map)

        model = FusionProtoClassifier(args=args, num_classes=len(fusion_label_map))
        model.to(args.device)

        trainer = train_model(model, train_loader, val_loader, args, class_weights=None)
        test_model(model, test_loader, args=args, trainer=trainer)
        _run_metrics_from_csv(args)
        raise SystemExit(0)

    # ----- all other stages --------------------------------------------------
    label_map = load_label_mappings(
        processed_dir=_processed_dir,
        mode=args.mode,
        label_set=args.label_set,
        custom_groups=args.custom_groups,
    )
    train_loader, val_loader, test_loader = get_dataloaders(args, label_map)

    class_weights = _make_class_weights(label_map)

    model = _build_proto_model(args, num_classes=len(label_map["all"]))
    model.to(args.device)

    trainer = train_model(model, train_loader, val_loader, args, class_weights)
    test_model(model, test_loader, args=args, trainer=trainer)
    _run_metrics_from_csv(args)