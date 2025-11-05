# /opt/gpudata/summereunann/echonext_suite/scripts/protoecgnet_original/proto_models2D.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from functools import reduce
import operator as op
from backbones import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

# deterministic mapping: all ranks will see the same thing
_BACKBONE_OUT_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}


def prototype_loss2d(
    logits,
    y_true,
    model,
    similarity_scores,
    class_weights,
    lam_clst,
    lam_sep,
    lam_spars,
    lam_div,
    lam_cnrst,
    use_contrastive=True,
):
    """
    Prototype-based loss function for ProtoECGNet (2D).
    Uses similarity scores that the model already computed.
    """

    device = model.prototype_vectors.device
    y_true = y_true.to(device).float()
    logits = logits.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    similarity_scores = similarity_scores.to(device)

    # (B, P, C)
    prototypes_of_correct_class = (
        model.prototype_class_identity.unsqueeze(0)
        .expand(y_true.shape[0], -1, -1)
        .to(device)
    )
    # mask with ground truth labels (multi-label)
    prototypes_of_correct_class = prototypes_of_correct_class * y_true.unsqueeze(1)

    # clustering: want correct-class prototypes to fire
    correct_class_prototype_activations, _ = torch.max(
        similarity_scores.unsqueeze(-1) * prototypes_of_correct_class, dim=1
    )

    # separation: want wrong-class prototypes to NOT fire
    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
    incorrect_class_prototype_activations, _ = torch.max(
        similarity_scores.unsqueeze(-1) * prototypes_of_wrong_class, dim=1
    )

    clst_loss = -torch.mean(correct_class_prototype_activations)
    sep_loss = torch.mean(incorrect_class_prototype_activations)

    # sparsity across prototypes
    spars_loss = torch.mean(torch.clamp(similarity_scores, min=0).sum(dim=1))

    # prototype diversity
    div_loss = model.get_prototype_orthogonalities()
    div_loss = div_loss / (model.num_prototypes ** 2)

    # contrastive part — only if we really have a cooccurrence matrix
    if use_contrastive and getattr(model, "label_cooccurrence", None) is not None:
        prototypes = model.prototype_vectors.view(model.num_prototypes, -1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        sim_matrix = torch.mm(prototypes_norm, prototypes_norm.T)
        sim_matrix = torch.clamp(sim_matrix, min=-1 + 1e-6, max=1 - 1e-6)

        P_assign = model.prototype_class_identity.float()
        L_cooccur = model.label_cooccurrence.to(device)

        pos_mask = torch.matmul(torch.matmul(P_assign, L_cooccur), P_assign.T)
        neg_mask = 1 - pos_mask

        pos_norm = pos_mask / (pos_mask.sum() + 1e-6)
        neg_norm = neg_mask / (neg_mask.sum() + 1e-6)

        pos_loss = torch.sum(sim_matrix * pos_norm)
        neg_loss = torch.sum(sim_matrix * neg_norm)

        cnrst_loss = -(pos_loss - neg_loss) / (model.num_prototypes ** 0.5)
    else:
        cnrst_loss = 0.0

    classification_loss = F.binary_cross_entropy_with_logits(
        logits, y_true, pos_weight=class_weights
    )

    return (
        classification_loss
        + lam_clst * clst_loss
        + lam_sep * sep_loss
        + lam_spars * spars_loss
        + lam_div * div_loss
        + lam_cnrst * cnrst_loss
    )


class ProtoECGNet2D(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        num_classes=5,
        single_class_prototype_per_class=5,
        joint_prototypes_per_border=0,
        proto_dim=512,
        proto_time_len=3,
        prototype_activation_function="log",
        latent_space_type="l2",
        add_on_layers_type="other",
        class_specific=False,
        last_layer_connection_weight=1.0,
        m=None,
        dropout=0.0,
        custom_groups=True,
        label_set="all",
        pretrained_weights=None,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # prototype layout
        # ------------------------------------------------------------------
        self.joint_prototypes_per_border = joint_prototypes_per_border
        self.single_class_prototype_per_class = single_class_prototype_per_class
        self.num_classes = num_classes

        num_class_pairs = self._ncr(num_classes, 2)
        self.num_prototypes = (
            self.single_class_prototype_per_class * self.num_classes
            + self.joint_prototypes_per_border * num_class_pairs
        )

        self.prototype_shape = (self.num_prototypes, proto_dim, 1, proto_time_len)
        self.proto_time_len = proto_time_len
        self.topk_k = 5

        self.prototype_activation_function = prototype_activation_function
        self.class_specific = class_specific
        self.epsilon = 1e-4
        self.custom_groups = custom_groups
        self.label_set = label_set
        self.m = m
        self.relu_on_cos = False
        self.input_vector_length = int(
            (self.prototype_shape[1] * self.prototype_shape[3]) ** 0.5
        )
        self.last_layer_connection_weight = last_layer_connection_weight
        self.latent_space_type = latent_space_type

        # EchoNext: we don't have the cooccurrence file, so disable it
        self.label_cooccurrence = None
        print("[INFO] Label co-occurrence disabled for EchoNext dataset")

        # ------------------------------------------------------------------
        # backbone
        # ------------------------------------------------------------------
        print(f"Initializing ProtoECGNet2D with backbone {backbone}...")
        self.backbone_name = backbone
        self.feature_extractor, backbone_only = self._get_backbone(
            backbone, num_classes, dropout, pretrained_weights
        )

        # decide how many channels the backbone emits
        self.backbone_out_channels = _BACKBONE_OUT_CHANNELS.get(backbone, proto_dim)

        # if the backbone itself already has a conv that reduces channels, respect it
        if hasattr(self.feature_extractor, "conv_reduce"):
            try:
                self.backbone_out_channels = self.feature_extractor.conv_reduce.out_channels
            except Exception:
                pass

        # strip classifiery things from the backbone
        if hasattr(self.feature_extractor, "flatten"):
            self.feature_extractor.flatten = nn.Identity()
        if hasattr(self.feature_extractor, "fc"):
            self.feature_extractor.fc = nn.Identity()

        # ------------------------------------------------------------------
        # prototypes + classifier
        # ------------------------------------------------------------------
        self.prototype_vectors = nn.Parameter(
            torch.randn(*self.prototype_shape), requires_grad=True
        )
        print(f"Initialized prototype_vectors with shape: {self.prototype_shape}")

        self.ones = nn.Parameter(
            torch.ones_like(self.prototype_vectors), requires_grad=False
        )
        self.classifier = nn.Linear(self.num_prototypes, num_classes, bias=False)

        # prototype → class assignment matrix
        self.register_buffer(
            "prototype_class_identity",
            self._create_prototype_labels(
                num_classes,
                single_class_prototype_per_class,
                joint_prototypes_per_border,
            ),
        )

        # ------------------------------------------------------------------
        # add-on layers: map backbone_out_channels → proto_dim
        # ------------------------------------------------------------------
        self._initialize_add_on_layers(
            add_on_layers_type,
            in_channels=self.backbone_out_channels,
            out_channels=proto_dim,
        )

        self._initialize_weights()

        # if we ever load a full model checkpoint, we mirror the original logic
        if backbone_only:
            print("Reset prototypes and classifier (backbone-only checkpoint).")
        elif pretrained_weights is not None:
            # same logic you had — load full model
            print(f"Now loading full model from {pretrained_weights}")
            ckpt = torch.load(
                pretrained_weights,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                weights_only=False,
            )
            state_dict = ckpt.get("state_dict", ckpt)

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "")
                new_state_dict[new_key] = v
            if "criterion.pos_weight" in new_state_dict:
                del new_state_dict["criterion.pos_weight"]

            if "prototype_vectors" in new_state_dict:
                loaded_prototypes = new_state_dict["prototype_vectors"].shape[0]
                print(
                    f"Checkpoint contains {loaded_prototypes} prototypes. Adjusting model initialization."
                )
                self.num_prototypes = loaded_prototypes
                self.prototype_shape = (
                    self.num_prototypes,
                    proto_dim,
                    1,
                    proto_time_len,
                )
                self.ones = nn.Parameter(
                    torch.ones(self.prototype_shape), requires_grad=False
                )
                self.prototype_vectors = nn.Parameter(
                    torch.randn(*self.prototype_shape), requires_grad=True
                )
                self.classifier = nn.Linear(
                    self.num_prototypes, self.num_classes, bias=False
                )
            self.load_state_dict(new_state_dict, strict=True)

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
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
        n_pairs = len(list(combinations(range(num_classes), 2)))
        identity_matrix = torch.zeros(
            (num_classes * single_class_prototype_per_class)
            + (n_pairs * joint_prototypes_per_border),
            num_classes,
        )

        # single-class prototypes
        for c in range(num_classes):
            for k in range(single_class_prototype_per_class):
                idx = c * single_class_prototype_per_class + k
                identity_matrix[idx, c] = 1

        # joint prototypes
        dual_start = num_classes * single_class_prototype_per_class
        for i, (c1, c2) in enumerate(combinations(range(num_classes), 2)):
            for k in range(joint_prototypes_per_border):
                idx = dual_start + i * joint_prototypes_per_border + k
                identity_matrix[idx, c1] = 1
                identity_matrix[idx, c2] = 1

        return identity_matrix

    def _initialize_add_on_layers(self, add_on_layers_type, in_channels, out_channels):
        # always project backbone_out_channels -> proto_dim
        if add_on_layers_type == "identity":
            self.add_on_layers = nn.Identity()
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                ),
                nn.Sigmoid(),
            )

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection()

    def set_last_layer_incorrect_connection(self, incorrect_class_connection=-0.5):
        if not hasattr(self, "prototype_class_identity"):
            raise ValueError(
                "[ERROR] prototype_class_identity not found. Initialize before calling."
            )

        correct_class_connection = (
            self.last_layer_connection_weight
            if self.last_layer_connection_weight is not None
            else 1.0
        )

        positive_mask = self.prototype_class_identity.T  # (C, P)
        negative_mask = 1 - positive_mask

        self.classifier.weight.data.copy_(
            correct_class_connection * positive_mask
            + incorrect_class_connection * negative_mask
        )

        print(
            f"Updated classifier weight matrix: Correct = {correct_class_connection}, Incorrect = {incorrect_class_connection}"
        )

    def _get_backbone(self, backbone, num_classes, dropout, pretrained_weights):
        backbones = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
            "resnet152": resnet152,
        }
        if backbone not in backbones:
            raise ValueError(f"Unsupported 2D backbone: {backbone}")

        model = backbones[backbone](num_classes=num_classes, dropout=dropout)
        backbone_only = False

        if pretrained_weights is not None:
            ckpt = torch.load(
                pretrained_weights,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                weights_only=False,
            )
            state_dict = ckpt.get("state_dict", ckpt)

            has_protos = any("prototype_vectors" in k for k in state_dict.keys())
            has_classifier = any("classifier" in k for k in state_dict.keys())

            if has_protos and has_classifier:
                print("Full ProtoECGNet2D checkpoint detected, leaving as is.")
                backbone_only = False
            else:
                print(
                    f"Loading backbone-only weights from {pretrained_weights} (prototypes & classifier stay)."
                )
                model.load_state_dict(state_dict, strict=False)
                backbone_only = True
        else:
            print("No pretrained weights provided")

        return model, backbone_only

    # ------------------------------------------------------------------
    # forward paths
    # ------------------------------------------------------------------
    def conv_features(self, x):
        x = self.feature_extractor(x)
        if not isinstance(self.add_on_layers, nn.Identity):
            x = self.add_on_layers(x)
        return x

    def prototype_distances(self, x, prototypes_of_wrong_class=None):
        conv_features = self.conv_features(x)

        if self.latent_space_type == "l2":
            distances = self._l2_convolution(conv_features)
            prototype_activations = self.distance_2_similarity(distances)
            return distances, prototype_activations

        elif self.latent_space_type == "arc":
            activations, marginless_activations = self.cos_activation(
                conv_features, prototypes_of_wrong_class
            )
            top_k, _ = torch.topk(activations, self.topk_k, dim=-1)
            pooled = F.avg_pool1d(top_k, kernel_size=top_k.shape[-1]).squeeze(-1)
            return None, pooled

        else:
            raise ValueError(f"Unsupported latent space type: {self.latent_space_type}")

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == "log":
            similarities = torch.log(
                (distances + 1) / (distances + self.epsilon)
            )  # (B, P, T)
        elif self.prototype_activation_function == "linear":
            similarities = -distances
        else:
            similarities = self.prototype_activation_function(distances)

        if self.topk_k > 1:
            topk_vals, _ = torch.topk(similarities, k=self.topk_k, dim=-1)
            similarities = torch.mean(topk_vals, dim=-1)
        else:
            similarities, _ = torch.max(similarities, dim=-1)

        return similarities  # (B, P)

    def forward(self, x, prototypes_of_wrong_class=None):
        features = self.conv_features(x)

        if self.latent_space_type == "l2":
            distances = self._l2_convolution(features)
            prototype_activations = self.distance_2_similarity(distances)
            logits = self.classifier(prototype_activations)
            return logits, distances, prototype_activations

        elif self.latent_space_type == "arc":
            activations, marginless_activations = self.cos_activation(
                features, prototypes_of_wrong_class
            )
            pooled = torch.mean(activations, dim=-1)
            pooled_marginless = torch.mean(marginless_activations, dim=-1)
            logits = self.classifier(pooled)
            marginless_logits = self.classifier(pooled_marginless)
            return logits, marginless_logits, pooled

    def push_forward(self, x):
        features = self.conv_features(x)
        if self.latent_space_type == "l2":
            distances = self._l2_convolution(features)
            activations = self.distance_2_similarity(distances)
            return features, activations
        else:
            activations, _ = self.cos_activation(features)
            return features, activations

    def _l2_convolution(self, x):
        # x: (B, C, 1, T) or (B, C, T)
        if x.dim() == 4:
            x = x.squeeze(2)
        B, C, T = x.shape
        num_windows = T - self.proto_time_len + 1

        distances = torch.zeros(
            (B, self.num_prototypes, num_windows), device=x.device
        )

        proto_flat = self.prototype_vectors.reshape(
            self.num_prototypes, C * self.proto_time_len
        )

        for i in range(num_windows):
            seg = x[:, :, i : i + self.proto_time_len]
            seg = seg.reshape(B, C * self.proto_time_len)
            distances[:, :, i] = torch.cdist(
                seg.unsqueeze(1), proto_flat.unsqueeze(0), p=2
            ).squeeze(1)

        return distances

    def cos_activation(self, x, prototypes_of_wrong_class=None):
        eps = self.epsilon
        input_vector_length = self.input_vector_length
        normalizing_factor = (self.prototype_shape[-2] * self.prototype_shape[-1]) ** 0.5

        B, C, _, T = x.shape
        L = self.proto_time_len
        num_windows = T - L + 1

        activations = torch.zeros((B, self.num_prototypes, num_windows), device=x.device)

        protos = self.prototype_vectors.squeeze(2)
        proto_norm = torch.sqrt(torch.sum(protos ** 2, dim=(1, 2), keepdim=True) + eps)
        protos_scaled = (protos / proto_norm) / normalizing_factor

        proto_flat = protos_scaled.view(self.num_prototypes, -1)

        for i in range(num_windows):
            seg = x[:, :, 0, i : i + L]
            seg_norm = torch.sqrt(torch.sum(seg ** 2, dim=(1, 2), keepdim=True) + eps)
            seg_scaled = (input_vector_length * seg / seg_norm) / normalizing_factor
            seg_flat = seg_scaled.view(B, -1)
            activations[:, :, i] = torch.matmul(seg_flat, proto_flat.T)

        return activations, activations

    def get_prototype_orthogonalities(self):
        P_flat = self.prototype_vectors.view(self.num_prototypes, -1)
        P_norm = F.normalize(P_flat, p=2, dim=1)
        sim = torch.matmul(P_norm, P_norm.T)
        eye = torch.eye(self.num_prototypes, device=self.prototype_vectors.device)
        off_diag = sim - eye
        return (off_diag ** 2).sum()

    def prune_prototypes(self, prototypes_to_prune):
        if not prototypes_to_prune:
            print("No prototypes to prune.")
            return

        keep = sorted(set(range(self.num_prototypes)) - set(prototypes_to_prune))
        self.prototype_vectors = nn.Parameter(
            self.prototype_vectors.data[keep], requires_grad=True
        )
        self.num_prototypes = len(keep)
        self.prototype_shape = list(self.prototype_vectors.size())

        old_w = self.classifier.weight.data[:, keep].clone()
        self.classifier = nn.Linear(
            self.num_prototypes, self.num_classes, bias=False
        ).to(self.prototype_vectors.device)
        with torch.no_grad():
            self.classifier.weight.copy_(old_w)

        if hasattr(self, "ones"):
            self.ones = nn.Parameter(self.ones.data[keep], requires_grad=False)

        self.prototype_class_identity = self.prototype_class_identity[keep, :]

        print(f"New prototype count: {self.num_prototypes}")

    def __repr__(self):
        return (
            "ProtoECGNet2D(\n"
            f"\tbackbone={self.backbone_name},\n"
            f"\tprototype_shape={self.prototype_shape},\n"
            f"\tnum_classes={self.num_classes},\n"
            f"\tepsilon={self.epsilon}\n"
            ")"
        )
