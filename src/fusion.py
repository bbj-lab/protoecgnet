import os
import torch
import pandas as pd
import torch.nn as nn
from ecg_utils import get_dataloaders, DATASET_PATH
from proto_models1D import ProtoECGNet1D
from proto_models2D import ProtoECGNet2D

class FusionProtoClassifier(nn.Module):
    def __init__(self, model_1d, model_2d_partial, model_2d_global, num_classes=71):
        super().__init__()
        self.model1d = model_1d.eval()
        self.model2d_partial = model_2d_partial.eval()
        self.model2d_global = model_2d_global.eval()

        for submodel in [self.model1d, self.model2d_partial, self.model2d_global]:
            for param in submodel.parameters():
                param.requires_grad = False

        self.num_classes = num_classes

        self.num_prototypes_1d = self.model1d.num_prototypes
        self.num_prototypes_2d_partial = self.model2d_partial.num_prototypes
        self.num_prototypes_2d_global = self.model2d_global.num_prototypes
        self.total_prototypes = self.num_prototypes_1d + self.num_prototypes_2d_partial + self.num_prototypes_2d_global

        self.classifier = nn.Linear(self.total_prototypes, num_classes, bias=True)

    def forward(self, x):
        x1d = x.squeeze(1)  # Convert [N, 1, 12, 1000] → [N, 12, 1000] for 1D branch only
        _, _, sim1d = self.model1d(x1d)
        _, _, sim2d_partial = self.model2d_partial(x)
        _, _, sim2d_global = self.model2d_global(x)

        sims = torch.cat([sim1d, sim2d_partial, sim2d_global], dim=1)
        logits = self.classifier(sims)
        return logits
        
# THE CODE BELOW IS NOT USED; instead, we call the normal dataloading code and use label_set='all' with custom_groups=False
def load_fusion_label_mappings():
    label_df = pd.read_csv(os.path.join(DATASET_PATH, "scp_statementsRegrouped2.csv"), index_col=0)
    assert "prototype_category" in label_df.columns

    labels_1 = label_df[label_df["prototype_category"] == 1].index.tolist()
    labels_3 = label_df[label_df["prototype_category"] == 3].index.tolist()
    labels_4 = label_df[label_df["prototype_category"] == 4].index.tolist()

    all_labels = sorted(set(labels_1 + labels_3 + labels_4))

    return {
        "1": labels_1,
        "3": labels_3,
        "4": labels_4,
        "all": all_labels
    }

def get_fusion_dataloaders(args, return_sample_ids=False):
    label_mappings = load_fusion_label_mappings()

    # Get input ECGs and full labels
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=args.batch_size,
        mode="2D",  # needed for shape (N, 1, 12, 1000)
        sampling_rate=args.sampling_rate,
        standardize=args.standardize,
        remove_baseline=args.remove_baseline,
        return_sample_ids=return_sample_ids,
        custom_groups=False,
        label_set="all",
        work_num=args.num_workers
    )

    def filter_labels(all_labels, target_group):
        target_scp_codes = label_mappings[target_group]
        indices = [i for i, name in enumerate(label_mappings["all"]) if name in target_scp_codes]
        return all_labels[:, indices]

    def collate_fn(batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)

        return inputs, {
            "1D": filter_labels(labels, "1"),
            "2D_partial": filter_labels(labels, "3"),
            "2D_global": filter_labels(labels, "4"),
            "full": labels
        }

    def wrap_loader(loader, shuffle=True):
        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )

    return (
        wrap_loader(train_loader, shuffle=True),
        wrap_loader(val_loader, shuffle=False),
        wrap_loader(test_loader, shuffle=False),
        class_weights
    )
