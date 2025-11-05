import os
from pathlib import Path

import torch
import pandas as pd
import torch.nn as nn

# NOTE: ecg_utils in repo doesn't export DATASET_PATH,
# so we only import get_dataloaders.
from ecg_utils import get_dataloaders

# These imports are in original file; keeping them
from proto_models1D import ProtoECGNet1D
from proto_models2D import ProtoECGNet2D


def _get_dataset_path() -> Path:
    """
    Figure out where the processed EchoNext data lives.

    Priority:
    1. Env var ECHONEXT_PROCESSED_DIR ( can set this before running)
    2. ../../processed relative to this file (repo layout)
    """
    env = os.environ.get("ECHONEXT_PROCESSED_DIR")
    if env:
        return Path(env)

    # fusion.py is in .../scripts/protoecgnet_original/
    # repo root should be two levels up
    return Path(__file__).resolve().parents[2] / "processed"


class FusionProtoClassifier(nn.Module):
    def __init__(self, model_1d, model_2d_partial, model_2d_global, num_classes=71):
        super().__init__()
        # freeze submodels and set to eval
        self.model1d = model_1d.eval()
        self.model2d_partial = model_2d_partial.eval()
        self.model2d_global = model_2d_global.eval()

        for submodel in [self.model1d, self.model2d_partial, self.model2d_global]:
            for param in submodel.parameters():
                param.requires_grad = False

        self.num_classes = num_classes

        # each submodel should expose num_prototypes
        self.num_prototypes_1d = self.model1d.num_prototypes
        self.num_prototypes_2d_partial = self.model2d_partial.num_prototypes
        self.num_prototypes_2d_global = self.model2d_global.num_prototypes
        self.total_prototypes = (
            self.num_prototypes_1d
            + self.num_prototypes_2d_partial
            + self.num_prototypes_2d_global
        )

        # final fusion head over concatenated similarities
        self.classifier = nn.Linear(self.total_prototypes, num_classes, bias=True)

    def forward(self, x):
        """
        x is expected to be 2D-shaped input: [N, 1, 12, 1000]
        - 1D branch wants [N, 12, 1000], so we squeeze channel dim
        - 2D branches can take x as-is
        """
        x1d = x.squeeze(1)  # [N, 1, 12, T] -> [N, 12, T] for 1D
        _, _, sim1d = self.model1d(x1d)
        _, _, sim2d_partial = self.model2d_partial(x)
        _, _, sim2d_global = self.model2d_global(x)

        # concat prototype similarities from all 3 models
        sims = torch.cat([sim1d, sim2d_partial, sim2d_global], dim=1)
        logits = self.classifier(sims)
        return logits


# THE CODE BELOW WAS MARKED "NOT USED" IN YOUR VERSION, but we keep it
# and make it safe so that simply importing fusion.py doesn't crash.
def load_fusion_label_mappings():
    dataset_path = _get_dataset_path()
    csv_path = dataset_path / "scp_statementsRegrouped2.csv"

    if not csv_path.exists():
        # don't crash on import — return a dummy structure
        return {
            "1": [],
            "3": [],
            "4": [],
            "all": [],
        }

    label_df = pd.read_csv(csv_path, index_col=0)
    assert "prototype_category" in label_df.columns

    labels_1 = label_df[label_df["prototype_category"] == 1].index.tolist()
    labels_3 = label_df[label_df["prototype_category"] == 3].index.tolist()
    labels_4 = label_df[label_df["prototype_category"] == 4].index.tolist()

    all_labels = sorted(set(labels_1 + labels_3 + labels_4))

    return {
        "1": labels_1,
        "3": labels_3,
        "4": labels_4,
        "all": all_labels,
    }


def get_fusion_dataloaders(args, return_sample_ids: bool = False):
    """
    Wrap normal dataloaders so that:
    - input is 2D-shaped (mode='2D')
    - labels are split into three groups based on the CSV
    """
    label_mappings = load_fusion_label_mappings()

    # normal ECG dataloading from   repo
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=args.batch_size,
        mode="2D",  # so we get shape (N, 1, 12, T)
        sampling_rate=args.sampling_rate,
        standardize=getattr(args, "standardize", False),
        remove_baseline=getattr(args, "remove_baseline", False),
        return_sample_ids=return_sample_ids,
        custom_groups=False,
        label_set="all",
        work_num=getattr(args, "num_workers", 0),
    )

    def filter_labels(all_labels, target_group):
        target_scp_codes = label_mappings.get(target_group, [])
        # label_mappings["all"] may be empty if CSV wasn't found
        all_codes = label_mappings.get("all", [])
        indices = [i for i, name in enumerate(all_codes) if name in target_scp_codes]
        if not indices:
            # return empty columns with correct batch size
            return all_labels[:, :0]
        return all_labels[:, indices]

    def collate_fn(batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)  # (B, 1, 12, T)
        labels = torch.stack(labels)  # (B, C)

        return inputs, {
            "1D": filter_labels(labels, "1"),
            "2D_partial": filter_labels(labels, "3"),
            "2D_global": filter_labels(labels, "4"),
            "full": labels,
        }

    def wrap_loader(loader, shuffle=True):
        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=getattr(args, "num_workers", 0),
            pin_memory=True,
        )

    return (
        wrap_loader(train_loader, shuffle=True),
        wrap_loader(val_loader, shuffle=False),
        wrap_loader(test_loader, shuffle=False),
        class_weights,
    )
