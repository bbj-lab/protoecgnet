#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proto_models1D.py
Extended to support custom 1D CNN-family backbones (simple/deep/wide, cnn_lstm, etc.)
with automatic channel adaptation and feature-dim inference.

This version is the long one (not the shortened one) and matches what
Lightning script expects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from functools import reduce
import operator as op

# ---------------------------------------------------------------------
# Sahil's ResNet backbones
# ---------------------------------------------------------------------
from backbones import (
    resnet1d18,
    resnet1d34,
    resnet1d50,
    resnet1d101,
    resnet1d152,
)

# ---------------------------------------------------------------------
# Custom CNN-family backbones
# ---------------------------------------------------------------------
from custom_backbones import (
    simple_cnn,
    deep_cnn,
    wide_cnn,
    cnn_gru,
    cnn_lstm,
    cnn_transformer,
    inception_cnn,
)

# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
def prototype_loss1d(
    logits,
    y_true,
    model,
    similarity_scores,
    lam_clst,
    lam_sep,
    lam_spars,
    lam_div,
    lam_cnrst,
    class_weights=None,
):
    device = logits.device
    y_true = y_true.to(device).float()
    similarity_scores = similarity_scores.to(device)

    # --- classification loss ---
    if class_weights is not None:
        class_weights = class_weights.to(device)
        cls_loss = F.binary_cross_entropy_with_logits(
            logits, y_true, pos_weight=class_weights
        )
    else:
        cls_loss = F.binary_cross_entropy_with_logits(logits, y_true)

    # if model has no prototype_vectors we just return classification
    if not hasattr(model, "prototype_vectors"):
        return cls_loss

    B = y_true.size(0)
    P = similarity_scores.size(1)

    proto_id = model.prototype_class_identity.to(device)
    proto_id_b = proto_id.unsqueeze(0).expand(B, -1, -1)  # (B, P, C)
    y_exp = y_true.unsqueeze(1)  # (B, 1, C)

    # positive/negative masks for each prototype
    correct_mask = proto_id_b * y_exp      # (B, P, C)
    wrong_mask = 1 - correct_mask

    sim_exp = similarity_scores.unsqueeze(-1)  # (B, P, 1)
    correct_act, _ = torch.max(sim_exp * correct_mask, dim=1)  # (B, C)
    wrong_act, _ = torch.max(sim_exp * wrong_mask, dim=1)      # (B, C)

    clst_loss = -correct_act.mean()
    sep_loss = wrong_act.mean()
    spars_loss = torch.clamp(similarity_scores, min=0).sum(dim=1).mean()

    # diversity among prototypes
    protos = model.prototype_vectors
    protos = F.normalize(protos.view(P, -1), p=2, dim=1)
    gram = protos @ protos.t()
    eye = torch.eye(P, device=device)
    div_loss = torch.norm(gram - eye, p="fro") ** 2 / (P ** 2)

    # contrastive optional
    cnrst_loss = torch.tensor(0.0, device=device)
    has_cooccur = getattr(model, "label_cooccurrence", None) is not None
    if has_cooccur and lam_cnrst > 0:
        L_cooccur = model.label_cooccurrence.to(device)
        P_assign = model.prototype_class_identity.float().to(device)
        pos_mask = P_assign @ L_cooccur @ P_assign.t()
        neg_mask = 1 - pos_mask
        pos_norm = pos_mask / (pos_mask.sum() + 1e-6)
        neg_norm = neg_mask / (neg_mask.sum() + 1e-6)
        sim_pp = protos @ protos.t()
        sim_pp = sim_pp.clamp(-1 + 1e-6, 1 - 1e-6)
        pos_loss = (sim_pp * pos_norm).sum()
        neg_loss = (sim_pp * neg_norm).sum()
        cnrst_loss = -(pos_loss - neg_loss) / (P ** 0.5)

    total = (
        cls_loss
        + lam_clst * clst_loss
        + lam_sep * sep_loss
        + lam_spars * spars_loss
        + lam_div * div_loss
        + lam_cnrst * cnrst_loss
    )
    return total


# ---------------------------------------------------------------------
# Channel adapter (defensive)
# ---------------------------------------------------------------------
class ChannelAdapter1D(nn.Module):
    """
    If datamodule gives (B, 1, L) but backbone expects (B, C, L) like 12,
    we project 1 -> C.
    If the input already has C channels, we just return it.
    """
    def __init__(self, in_ch=1, out_ch=12):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, L)
        if x.size(1) == self.proj.out_channels:
            # already correct #channels
            return x
        return self.proj(x)


# ---------------------------------------------------------------------
# ProtoECGNet1D
# ---------------------------------------------------------------------
class ProtoECGNet1D(nn.Module):
    def __init__(
        self,
        backbone="resnet1d18",
        num_classes=5,
        single_class_prototype_per_class=5,
        joint_prototypes_per_border=2,
        proto_dim=512,
        prototype_activation_function="log",
        latent_space_type="l2",
        add_on_layers_type="linear",
        class_specific=False,
        last_layer_connection_weight=1.0,
        m=None,
        dropout=0.0,
        custom_groups=True,
        label_set="all",
        pretrained_weights=None,
    ):
        super().__init__()
        self.joint_prototypes_per_border = joint_prototypes_per_border
        self.single_class_prototype_per_class = single_class_prototype_per_class
        self.num_classes = num_classes

        # total prototypes = per-class + joint
        num_class_pairs = self._ncr(num_classes, 2)
        self.num_prototypes = (
            self.single_class_prototype_per_class * self.num_classes
            + self.joint_prototypes_per_border * num_class_pairs
        )
        self.prototype_shape = (self.num_prototypes, proto_dim)

        self.prototype_activation_function = prototype_activation_function
        self.class_specific = class_specific
        self.epsilon = 1e-4
        self.m = m
        self.custom_groups = custom_groups
        self.label_set = label_set
        self.relu_on_cos = False
        self.last_layer_connection_weight = last_layer_connection_weight
        self.latent_space_type = latent_space_type

        # author sometimes had label co-occurrence; for EchoNext you disabled it
        self.label_cooccurrence = None
        print("[INFO] Label co-occurrence disabled for EchoNext dataset")

        # --------------------------------------------------------------
        # 1) build backbone (author + custom)
        # --------------------------------------------------------------
        print(f"Initializing ProtoECGNet1D with backbone {backbone}...")
        backbone_model, backbone_only = self._get_backbone(
            backbone, num_classes, dropout, pretrained_weights
        )

        # --------------------------------------------------------------
        # 2) maybe wrap with channel adapter (1 -> needed_channels)
        # --------------------------------------------------------------
        backbone_model = self._maybe_wrap_channel_adapter(backbone_model)

        # resnets have .fc, zero it out
        self.feature_extractor = backbone_model
        if hasattr(self.feature_extractor, "fc"):
            self.feature_extractor.fc = nn.Identity()

        # --------------------------------------------------------------
        # 3) infer backbone feature dim from dummy forward
        # --------------------------------------------------------------
        backbone_feat_dim = self._infer_backbone_feat_dim(self.feature_extractor)

        # --------------------------------------------------------------
        # 4) create prototype vectors + classifier
        # --------------------------------------------------------------
        self.prototype_vectors = nn.Parameter(
            torch.randn(*self.prototype_shape), requires_grad=True
        )
        self.ones = nn.Parameter(
            torch.ones_like(self.prototype_vectors), requires_grad=False
        )
        self.classifier = nn.Linear(self.num_prototypes, num_classes, bias=False)

        # prototype -> label assignment matrix
        self.register_buffer(
            "prototype_class_identity",
            self._create_prototype_labels(
                num_classes,
                single_class_prototype_per_class,
                joint_prototypes_per_border,
            ),
        )

        # --------------------------------------------------------------
        # 5) add-on layers to map backbone feat -> proto_dim
        # --------------------------------------------------------------
        self._initialize_add_on_layers(add_on_layers_type, backbone_feat_dim, proto_dim)

        # --------------------------------------------------------------
        # 6) init weights + classifier connections
        # --------------------------------------------------------------
        self._initialize_weights()

        if backbone_only:
            print("Reset prototypes and classifier (backbone-only checkpoint detected).")

    # --------------------------------------------------------------
    @staticmethod
    def _ncr(n, r):
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer // denom

    def _create_prototype_labels(
        self,
        num_classes,
        single_class_prototype_per_class,
        joint_prototypes_per_border,
    ):
        pairs = list(combinations(range(num_classes), 2))
        P = num_classes * single_class_prototype_per_class + len(pairs) * joint_prototypes_per_border
        mat = torch.zeros(P, num_classes)
        # per-class protos
        for c in range(num_classes):
            for k in range(single_class_prototype_per_class):
                idx = c * single_class_prototype_per_class + k
                mat[idx, c] = 1
        # joint protos
        start = num_classes * single_class_prototype_per_class
        for i, (c1, c2) in enumerate(pairs):
            for k in range(joint_prototypes_per_border):
                idx = start + i * joint_prototypes_per_border + k
                mat[idx, c1] = 1
                mat[idx, c2] = 1
        return mat

    # --------------------------------------------------------------
    # backbone registry
    # --------------------------------------------------------------
    def _get_backbone(self, backbone, num_classes, dropout, pretrained_weights):
        backbones = {
            # Author
            "resnet1d18": resnet1d18,
            "resnet1d34": resnet1d34,
            "resnet1d50": resnet1d50,
            "resnet1d101": resnet1d101,
            "resnet1d152": resnet1d152,
            # Custom
            "simple_cnn": simple_cnn,
            "deep_cnn": deep_cnn,
            "wide_cnn": wide_cnn,
            "cnn_gru": cnn_gru,
            "cnn_lstm": cnn_lstm,
            "cnn_transformer": cnn_transformer,
            "inception_cnn": inception_cnn,
        }

        if backbone not in backbones:
            raise ValueError(f"Unsupported 1D backbone: {backbone}")

        model = backbones[backbone](num_classes=num_classes, dropout=dropout)
        backbone_only = False

        if pretrained_weights is not None:
            ckpt = torch.load(
                pretrained_weights,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            has_proto = any("prototype_vectors" in k for k in state.keys())
            has_clf = any("classifier" in k for k in state.keys())
            if not (has_proto or has_clf):
                print(f"Loading backbone-only weights from {pretrained_weights}")
                model.load_state_dict(state, strict=False)
                backbone_only = True
            else:
                print(f"[INFO] Loaded full pretrained model weights from {pretrained_weights}")
        else:
            print("No pretrained weights provided")

        return model, backbone_only

    # --------------------------------------------------------------
    def _maybe_wrap_channel_adapter(self, backbone_model):
        """
        Detect the first Conv1d in the backbone. If it wants >1 channel,
        wrap a 1->C adapter in front. The adapter itself is defensive,
        so if the batch already has C channels it will just pass through.
        """
        in_ch_needed = None
        for _, mod in backbone_model.named_modules():
            if isinstance(mod, nn.Conv1d):
                in_ch_needed = mod.in_channels
                break

        if in_ch_needed is None:
            # backbone has no conv1d? ok
            return backbone_model

        if in_ch_needed > 1:
            print(f"[INFO] Adding channel adapter: 1 -> {in_ch_needed}")
            return nn.Sequential(
                ChannelAdapter1D(1, in_ch_needed),
                backbone_model,
            )

        return backbone_model

    # --------------------------------------------------------------
    def _infer_backbone_feat_dim(self, backbone):
        """
        Run a dummy through the backbone to find feature dim.
        Assumes ECG length ~2500 (fits 500Hz, 5s example).
        """
        device = next(backbone.parameters()).device

        # figure out how many channels this *wrapped* backbone wants
        # for nn.Sequential(adapter, backbone) the first real module is index 1
        modules = list(backbone.modules())
        # default
        in_ch = 1
        for m in modules:
            if isinstance(m, nn.Conv1d):
                in_ch = m.in_channels
                break

        dummy = torch.randn(1, in_ch, 2500, device=device)
        with torch.no_grad():
            feats = backbone(dummy)

        # some custom CNNs return (B, C, T); pool over T
        if feats.dim() == 3:
            feats = feats.mean(dim=-1)

        feat_dim = feats.shape[1]
        return feat_dim

    # --------------------------------------------------------------
    def _initialize_add_on_layers(self, add_on_layers_type, in_dim, proto_dim):
        if add_on_layers_type == "identity":
            self.add_on_layers = nn.Identity()
        else:
            # simple MLP
            self.add_on_layers = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, proto_dim),
                nn.Sigmoid(),
            )
        print(f"[INFO] Built add_on_layers with input dim {in_dim} -> proto_dim {proto_dim}")

    # --------------------------------------------------------------
    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection()

    # --------------------------------------------------------------
    def set_last_layer_incorrect_connection(self, incorrect_class_connection=-0.5):
        if not hasattr(self, "prototype_class_identity"):
            raise ValueError("prototype_class_identity missing")
        correct = self.last_layer_connection_weight
        pos_mask = self.prototype_class_identity.T  # (C, P)
        neg_mask = 1 - pos_mask
        self.classifier.weight.data.copy_(
            correct * pos_mask + incorrect_class_connection * neg_mask
        )
        print(
            f"Updated classifier weight matrix: "
            f"Correct = {correct}, Incorrect = {incorrect_class_connection}"
        )

    # --------------------------------------------------------------
    # Forward stuff
    # --------------------------------------------------------------
    def conv_features(self, x):
        feats = self.feature_extractor(x)
        if feats.dim() == 3:
            # (B, C, T) -> (B, C)
            feats = feats.mean(dim=-1)
        feats = self.add_on_layers(feats)
        return feats

    def _l2_convolution(self, feats):
        return torch.cdist(feats, self.prototype_vectors, p=2)

    def distance_2_similarity(self, dists):
        if self.prototype_activation_function == "log":
            return torch.log((dists + 1.0) / (dists + self.epsilon))
        elif self.prototype_activation_function == "linear":
            return -dists
        else:
            # assume it's a callable
            return self.prototype_activation_function(dists)

    def forward(self, x, prototypes_of_wrong_class=None):
        feats = self.conv_features(x)
        if self.latent_space_type == "l2":
            dists = self._l2_convolution(feats)
            activations = self.distance_2_similarity(dists)
            logits = self.classifier(activations)
            return logits, dists, activations
        else:
            feats_n = F.normalize(feats, p=2, dim=1)
            protos_n = F.normalize(self.prototype_vectors, p=2, dim=1)
            acts = feats_n @ protos_n.t()
            logits = self.classifier(acts)
            return logits, acts, acts

    def push_forward(self, x):
        feats = self.conv_features(x)
        dists = self._l2_convolution(feats)
        acts = self.distance_2_similarity(dists)
        return feats, acts