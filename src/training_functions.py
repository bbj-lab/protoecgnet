# ==== TORCHAUDIO STUB TO AVOID BROKEN BINARY LOADS ====
# This MUST come before importing pytorch_lightning / torchmetrics.
import sys
import types
import importlib.machinery

ta = types.ModuleType("torchaudio")
ta.__file__ = "<dummy-torchaudio>"
ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
sys.modules["torchaudio"] = ta

for sub in (
    "_extension",
    "functional",
    "transforms",
    "datasets",
    "_extension.utils",
):
    name = f"torchaudio.{sub}"
    m = types.ModuleType(name)
    m.__file__ = "<dummy-torchaudio>"
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = m

    parts = sub.split(".")
    parent = ta
    for p in parts[:-1]:
        if not hasattr(parent, p):
            submod_name = f"torchaudio.{p}"
            submod = types.ModuleType(submod_name)
            submod.__file__ = "<dummy-torchaudio>"
            submod.__spec__ = importlib.machinery.ModuleSpec(submod_name, None)
            sys.modules[submod_name] = submod
            setattr(parent, p, submod)
        parent = getattr(parent, p)
    setattr(parent, parts[-1], m)
# ==== END TORCHAUDIO STUB ====


from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    y = y_true.astype(int)
    if y.min() == y.max():
        return None
    try:
        return float(roc_auc_score(y, y_prob))
    except Exception:
        return None

def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    y = y_true.astype(int)
    if y.min() == y.max():
        return None
    try:
        return float(average_precision_score(y, y_prob))
    except Exception:
        return None

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    auroc = _safe_auroc(y_true, y_prob)
    auprc = _safe_auprc(y_true, y_prob)
    thr_best = f1_best = None
    try:
        p, r, t = precision_recall_curve(y_true.astype(int), y_prob)
        if len(t) > 0:
            f1s = 2 * (p[1:] * r[1:]) / (p[1:] + r[1:] + 1e-12)
            idx = int(np.nanargmax(f1s))
            f1_best = float(f1s[idx])
            thr_best = float(t[idx])
    except Exception:
        pass
    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": f1_best,
        "best_thr": thr_best,
    }

def multilabel_metrics(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict:
    assert Y_true.shape == Y_prob.shape
    N, L = Y_true.shape
    if valid_mask is None:
        valid_mask = np.ones(L, dtype=bool)

    per_label = []
    aurocs, auprcs = [], []
    for j in range(L):
        yj = Y_true[:, j]
        pj = Y_prob[:, j]
        aj = _safe_auroc(yj, pj) if valid_mask[j] else None
        bj = _safe_auprc(yj, pj) if valid_mask[j] else None
        per_label.append({"auroc": aj, "auprc": bj})
        if aj is not None:
            aurocs.append(aj)
        if bj is not None:
            auprcs.append(bj)

    macro_auroc = float(np.mean(aurocs)) if len(aurocs) else None
    macro_auprc = float(np.mean(auprcs)) if len(auprcs) else None

    if valid_mask.any():
        y_flat = Y_true[:, valid_mask].reshape(-1)
        p_flat = Y_prob[:, valid_mask].reshape(-1)
        micro_auroc = _safe_auroc(y_flat, p_flat)
        micro_auprc = _safe_auprc(y_flat, p_flat)
    else:
        micro_auroc = micro_auprc = None

    return {
        "per_label": per_label,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
    }

def compute_metrics_dict(mode: str, y_true, y_prob, valid_mask: Optional[np.ndarray] = None) -> Dict:
    m = mode.lower()
    if m == "binary":
        return binary_metrics(y_true, y_prob)
    else:
        return multilabel_metrics(y_true, y_prob, valid_mask=valid_mask)



import os
import torch
import pytorch_lightning as pl
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import (
    multilabel_confusion_matrix,
)

from proto_models1D import ProtoECGNet1D, prototype_loss1d
from proto_models2D import ProtoECGNet2D, prototype_loss2d
from fusion import FusionProtoClassifier
from push import push_prototypes1d, push_prototypes2d
import torch.nn.functional as F
from collections import defaultdict


def get_class_names(args):
    from ecg_utils import load_label_mappings

    label_mappings = load_label_mappings(
        custom_groups=args.custom_groups,
        prototype_category=int(args.label_set) if args.custom_groups else None,
    )

    if args.custom_groups:
        return label_mappings["custom"]
    else:
        return label_mappings[args.label_set]


def seed_everything(seed):
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_study(study, filename):
    joblib.dump(study, filename)


def load_study(filename):
    return joblib.load(filename)


def save_model_weights(model, job_name, stage, save_dir="./checkpoints", save_weights=True):
    if save_weights:
        save_prepath = os.path.join(save_dir, job_name)
        os.makedirs(save_prepath, exist_ok=True)
        save_path = os.path.join(save_prepath, f"{job_name}_{stage}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")


def load_model_weights(model, checkpoint_path):
    if checkpoint_path:
        state_dict = torch.load(
            checkpoint_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {checkpoint_path}")


class ECGTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3, l2=0, args=None, class_weights=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.l2 = l2
        self.args = args
        self.is_proto = isinstance(model, (ProtoECGNet1D, ProtoECGNet2D, FusionProtoClassifier))
        self.class_weights = class_weights

        self.train_preds, self.train_labels = [], []
        self.val_preds, self.val_labels = [], []
        self.test_preds = []
        self.test_labels = []
        self.test_probs = []
        self.class_names = get_class_names(args)
        self.num_classes = len(self.class_names)

        self.save_hyperparameters(ignore=["model", "class_weights"])

    # ----------------- helpers -----------------
    def compute_l1_loss(self):
        if hasattr(self.model, "prototype_class_identity"):
            l1_mask = 1 - torch.t(self.model.prototype_class_identity).to(self.device)
            l1_penalty = (self.model.classifier.weight * l1_mask).norm(p=1)
            return l1_penalty
        return torch.norm(self.model.classifier.weight, p=1)

    def _build_loss(self, logits, y, similarity_scores=None):
        cw = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if self.args.training_stage in ["classifier", "fusion"]:
            bce = torch.nn.BCEWithLogitsLoss(pos_weight=cw)(logits, y.to(logits.device))
            return bce + self.args.l1 * self.compute_l1_loss()

        if self.args.training_stage in ["prototypes", "joint"]:
            if self.args.dimension == "1D":
                return prototype_loss1d(
                    logits=logits,
                    y_true=y,
                    model=self.model,
                    similarity_scores=similarity_scores,
                    lam_clst=self.args.lam_clst,
                    lam_sep=self.args.lam_sep,
                    lam_spars=self.args.lam_spars,
                    lam_div=self.args.lam_div,
                    lam_cnrst=self.args.lam_cnrst,
                    class_weights=cw,
                )
            else:
                return prototype_loss2d(
                    logits=logits,
                    y_true=y,
                    model=self.model,
                    similarity_scores=similarity_scores,
                    lam_clst=self.args.lam_clst,
                    lam_sep=self.args.lam_sep,
                    lam_spars=self.args.lam_spars,
                    lam_div=self.args.lam_div,
                    lam_cnrst=self.args.lam_cnrst,
                    class_weights=cw,
                )

        bce = torch.nn.BCEWithLogitsLoss(pos_weight=cw)(logits, y.to(logits.device))
        return bce

    # ----------------- Lightning hooks -----------------
    def forward(self, x):
        output = self.model(x)
        if self.args.training_stage in ["prototypes", "joint"]:
            logits, distances, similarity_scores = output
        else:
            logits = output
            distances = None
            similarity_scores = None
        return logits, distances, similarity_scores

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        if isinstance(y, dict):
            y = y["full"]

        output = self.model(x)
        if self.args.training_stage in ["feature_extractor", "fusion"]:
            logits = output
            similarity_scores = None
        elif self.args.training_stage in ["prototypes", "joint"]:
            logits, _, similarity_scores = output
        elif self.args.training_stage == "classifier":
            logits, _, _ = output
            similarity_scores = None
        else:
            logits = output
            similarity_scores = None

        loss = self._build_loss(logits, y, similarity_scores)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.train_preds.append(probs)
        self.train_labels.append(y.detach().cpu().numpy())
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        if len(self.train_preds) == 0:
            return

        all_preds = np.concatenate(self.train_preds, axis=0)
        all_labels = np.concatenate(self.train_labels, axis=0)

        m = compute_metrics_dict("multilabel", all_labels, all_preds)
        auc_for_log = m["macro_auroc"] if m["macro_auroc"] is not None else m["micro_auroc"]
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average="micro")

        if auc_for_log is not None:
            self.log("train_auc", auc_for_log, prog_bar=True, sync_dist=True)
        else:
            self.log("train_auc", float("nan"), prog_bar=True, sync_dist=True)

        self.log("train_f1", f1, prog_bar=True, sync_dist=True)

        self.train_preds, self.train_labels = [], []

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        if isinstance(y, dict):
            y = y["full"]

        output = self.model(x)
        if self.args.training_stage in ["feature_extractor", "fusion"]:
            logits = output
            similarity_scores = None
        elif self.args.training_stage in ["prototypes", "joint"]:
            logits, _, similarity_scores = output
        elif self.args.training_stage == "classifier":
            logits, _, _ = output
            similarity_scores = None
        else:
            logits = output
            similarity_scores = None

        loss = self._build_loss(logits, y, similarity_scores)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.val_preds.append(probs)
        self.val_labels.append(y.detach().cpu().numpy())
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return

        all_preds = np.concatenate(self.val_preds, axis=0)
        all_labels = np.concatenate(self.val_labels, axis=0)

        m = compute_metrics_dict("multilabel", all_labels, all_preds)
        val_auc = m["macro_auroc"] if m["macro_auroc"] is not None else m["micro_auroc"]
        val_pr = m["macro_auprc"] if m["macro_auprc"] is not None else m["micro_auprc"]

        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average="micro")

        if val_auc is not None:
            self.log("val_auc", val_auc, prog_bar=True, sync_dist=True)
        else:
            self.log("val_auc", float("nan"), prog_bar=True, sync_dist=True)

        if val_pr is not None:
            self.log("val_auprc", val_pr, prog_bar=False, sync_dist=True)

        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

        self.val_preds, self.val_labels = [], []

    def test_step(self, batch, batch_idx=None):
        x, y = batch
        if isinstance(y, dict):
            y = y["full"]

        output = self.model(x)
        if self.args.training_stage in ["feature_extractor", "fusion"]:
            logits = output
            similarity_scores = None
        elif self.args.training_stage in ["prototypes", "joint"]:
            logits, _, similarity_scores = output
        elif self.args.training_stage == "classifier":
            logits, _, _ = output
            similarity_scores = None
        else:
            logits = output
            similarity_scores = None

        loss = self._build_loss(logits, y, similarity_scores)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        self.test_preds.append(preds.cpu().numpy())
        self.test_labels.append(y.cpu().numpy())
        self.test_probs.append(probs.cpu().numpy())

        return {
            "test_loss": loss,
        }

    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return

        all_preds = np.concatenate(self.test_preds, axis=0)
        all_labels = np.concatenate(self.test_labels, axis=0)
        all_probs = np.concatenate(self.test_probs, axis=0)

        assert all_preds.shape == all_labels.shape == all_probs.shape, "Shape mismatch in test outputs"

        df_dict = {"ID": np.arange(len(all_preds))}
        for i, class_name in enumerate(self.class_names):
            df_dict[f"Label_{class_name}"] = all_labels[:, i]
            df_dict[f"Pred_{class_name}"] = all_preds[:, i]
            df_dict[f"Prob_{class_name}"] = all_probs[:, i]

        version = self.logger.version if self.logger else "0"
        save_dir = os.path.join(self.args.test_dir, self.args.job_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.args.job_name}_test_results_v{version}.csv")
        df = pd.DataFrame(df_dict)
        df.to_csv(save_path, index=False)
        print(f" Saved test predictions to {save_path}")

        # ---- SAFE METRICS ON TEST ----
        m = compute_metrics_dict("multilabel", all_labels, all_probs)
        macro_auc = m["macro_auroc"]
        macro_pr = m["macro_auprc"]
        micro_auc = m["micro_auroc"]
        micro_pr = m["micro_auprc"]

        if macro_auc is not None:
            self.log("test_macro_auroc", macro_auc, prog_bar=True, sync_dist=True)
        if macro_pr is not None:
            self.log("test_macro_auprc", macro_pr, prog_bar=False, sync_dist=True)
        if micro_auc is not None:
            self.log("test_micro_auroc", micro_auc, prog_bar=False, sync_dist=True)
        if micro_pr is not None:
            self.log("test_micro_auprc", micro_pr, prog_bar=False, sync_dist=True)

        # confusion matrix
        cm = multilabel_confusion_matrix(all_labels, all_preds)

        if self.logger:
            fig_cm = self.plot_confusion_matrix(cm)
            fig_roc = self.safe_plot_roc_curve(all_labels, all_probs, m["per_label"])
            fig_pr = self.safe_plot_pr_curve(all_labels, all_probs, m["per_label"])

            self.logger.experiment.add_figure("Test/Confusion_Matrix", fig_cm, global_step=self.current_epoch)
            self.logger.experiment.add_figure("Test/ROC_Curve", fig_roc, global_step=self.current_epoch)
            self.logger.experiment.add_figure("Test/PR_Curve", fig_pr, global_step=self.current_epoch)

        self.test_preds = []
        self.test_labels = []
        self.test_probs = []

    # ----------------- plotting -----------------
    def plot_confusion_matrix(self, cm):
        num_classes = cm.shape[0]
        fig, axes = plt.subplots(1, num_classes, figsize=(max(15, 3*num_classes), 5))
        if num_classes == 1:
            axes = [axes]
        for i, class_name in enumerate(self.class_names):
            sns.heatmap(cm[i], annot=True, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(class_name)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
        plt.tight_layout()
        return fig

    def safe_plot_roc_curve(self, y_true, y_probs, per_label_metrics):
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            if per_label_metrics[i]["auroc"] is None:
                continue
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            auc_score = per_label_metrics[i]["auroc"]
            ax.plot(fpr, tpr, label=f"{class_name} (AUC = {auc_score:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (valid labels only)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig

    def safe_plot_pr_curve(self, y_true, y_probs, per_label_metrics):
        from sklearn.metrics import precision_recall_curve
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            if per_label_metrics[i]["auprc"] is None:
                continue
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            pr_auc = per_label_metrics[i]["auprc"]
            ax.plot(recall, precision, label=f"{class_name} (AP = {pr_auc:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves (valid labels only)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        return fig

    # ----------------- optim -----------------
    def configure_optimizers(self):
        if self.args.training_stage == "prototypes":
            optimizer = torch.optim.Adam([self.model.prototype_vectors], lr=self.lr, weight_decay=self.l2)
        elif self.args.training_stage in ["classifier", "fusion"]:
            optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.lr, weight_decay=self.l2)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)

        if self.args.scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        elif self.args.scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler_type == "CyclicLR":
            base_lr = self.lr * 0.1
            max_lr = self.lr * 10
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=2000, mode="triangular"
            )
        else:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def train_model(base_model, train_loader, val_loader, args, class_weights, trainer=None):
    model = ECGTrainer(base_model, lr=args.lr, l2=args.l2, args=args, class_weights=class_weights)

    print(f"Starting training for job: {args.job_name}")
    print("Initializing TensorBoard Logger...")

    logger = TensorBoardLogger(args.log_dir, name=args.job_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, args.job_name),
        filename="{epoch}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(monitor="val_auc", patience=args.patience, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    print("Initializing PyTorch Lightning Trainer...")

    if args.resume_checkpoint:
        print(f"Resuming from checkpoint...")
    else:
        print(f"Not using a checkpoint...")

    if trainer is None:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            strategy="auto",
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        )

    if args.training_stage == "feature_extractor":
        print("Training feature extractor...")
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)

    elif args.training_stage == "prototypes":
        print("Training add-on layers and prototype layer...")
        for param in model.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in model.model.classifier.parameters():
            param.requires_grad = False
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)

    elif args.training_stage == "joint":
        print("Joint training of feature extractor, add-on layers, and prototype layer (without classifier)...")
        for param in model.model.feature_extractor.parameters():
            param.requires_grad = True
        for param in model.model.classifier.parameters():
            param.requires_grad = False
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)

    elif args.training_stage == "projection":
        print("Performing prototype projection...")
        proj_dir = os.path.join(args.checkpoint_dir, args.job_name)
        os.makedirs(proj_dir, exist_ok=True)
        if args.dimension == "1D":
            push_prototypes1d(
                model.model,
                train_loader,
                proj_dir,
                args.label_set,
                args.job_name,
                logger,
                device=args.device,
                custom_groups=args.custom_groups,
            )
        elif args.dimension == "2D":
            push_prototypes2d(
                model.model,
                train_loader,
                proj_dir,
                args.label_set,
                args.job_name,
                logger,
                device=args.device,
                custom_groups=args.custom_groups,
            )
        save_model_weights(model, args.job_name, "projection", args.checkpoint_dir, args.save_weights)

    elif args.training_stage == "classifier":
        print("Fine-tuning classifier (classifier-only training)...")
        for param in model.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in model.model.add_on_layers.parameters():
            param.requires_grad = False
        with torch.no_grad():
            model.model.prototype_vectors = torch.nn.Parameter(
                model.model.prototype_vectors.detach(), requires_grad=False
            )
        for param in model.model.classifier.parameters():
            param.requires_grad = True
        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)

    elif args.training_stage == "fusion":
        print("Training fusion classifier using similarity scores from pretrained models...")

        for submodel in [model.model.model1d, model.model.model2d_partial, model.model.model2d_global]:
            submodel.eval()
            for param in submodel.parameters():
                param.requires_grad = False

        with torch.no_grad():
            model.model.model1d.prototype_vectors = torch.nn.Parameter(
                model.model.model1d.prototype_vectors.detach(), requires_grad=False
            )
            model.model.model2d_partial.prototype_vectors = torch.nn.Parameter(
                model.model.model2d_partial.prototype_vectors.detach(), requires_grad=False
            )
            model.model.model2d_global.prototype_vectors = torch.nn.Parameter(
                model.model.model2d_global.prototype_vectors.detach(), requires_grad=False
            )

        for param in model.model.classifier.parameters():
            param.requires_grad = True

        trainer.fit(model, train_loader, val_loader, ckpt_path="last" if args.resume_checkpoint else None)

    else:
        raise ValueError("Invalid training stage specified")

    hparams = vars(args)
    final_metrics = trainer.callback_metrics
    logger.log_hyperparams(hparams, final_metrics)

    return trainer


def test_model(base_model, test_loader, args, device="cuda", trainer=None):
    model = ECGTrainer(base_model, args=args).to(device)
    logger = TensorBoardLogger(args.log_dir, name=args.job_name)

    if trainer is None:
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            logger=logger,
        )
        print("\n--- Evaluating on Test Set ---")
        test_results = trainer.test(model, dataloaders=test_loader)

    else:
        best_ckpt_path = None
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint) and callback.best_model_path:
                best_ckpt_path = callback.best_model_path
                break

        if best_ckpt_path and os.path.exists(best_ckpt_path):
            print("\n--- Evaluating Best Epoch Weights on Test Set ---")
            state_dict = torch.load(best_ckpt_path, map_location=device, weights_only=False)["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            test_results = trainer.test(model, dataloaders=test_loader)
        else:
            print("\n--- No Best Checkpoint Found, Using Current Model Weights ---")
            test_results = trainer.test(model, dataloaders=test_loader)

    print(f"Test Metrics: {test_results}")

    return test_results
