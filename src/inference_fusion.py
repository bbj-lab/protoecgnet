"""
ProtoECGNet Fusion Inference Script

This script provides standalone inference capabilities for trained ProtoECGNet fusion models
that combine 1D rhythm, 2D partial morphology, and 2D global branches.

Usage:
    python inference_fusion.py --target-ecg 65 # will do inference on only ECG 65 (ensure that you choose a test ECG)
    python inference_fusion.py --num-samples -1 # will do inference on the full test set if you use -1, otherwise you can specify a certain number of ECGs (this argument is overridden if you specify a target_ecg)
"""

import torch
import numpy as np
import json
import os
import argparse
import pandas as pd
import ast
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D
from proto_models2D import ProtoECGNet2D
from fusion import FusionProtoClassifier, load_fusion_label_mappings, get_fusion_dataloaders
from training_functions import seed_everything
import pprint

# Configuration - modify as necessary
# Fusion Model Configuration
FUSION_WEIGHTS = "/path/to/fusion_weights.ckpt"

# 1D
WEIGHTS_1D = "/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat1_new_redo1_proj1/cat1_new_redo1_proj1_projection.pth"
METADATA_1D = "/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat1_new_redo1_proj1/cat1_new_redo1_proj1_prototype_metadata.json"
DIMENSION_1D = "1D"
LABEL_SET_1D = "1"
PROTO_TIME_LEN_1D = 32
PROTOTYPES_PER_CLASS_1D = 5
BACKBONE_1D = "resnet1d18"
# 2D partial
WEIGHTS_2D_PARTIAL = "/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat3_2D_redo1_tunejoint1_trial96_proj/cat3_2D_redo1_tunejoint1_trial96_proj_projection.pth"
METADATA_2D_PARTIAL = "/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat3_2D_redo1_tunejoint1_trial96_proj/cat3_2D_redo1_tunejoint1_trial96_proj_prototype_metadata.json"
DIMENSION_2D_PARTIAL = "2D"
LABEL_SET_2D_PARTIAL = "3"
PROTO_TIME_LEN_2D_PARTIAL = 3
PROTOTYPES_PER_CLASS_2D_PARTIAL = 18
BACKBONE_2D_PARTIAL = "resnet18"
# 2D global
WEIGHTS_2D_GLOBAL = "/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat4_2D_redo1_tunejoint1_trial5_proj/cat4_2D_redo1_tunejoint1_trial5_proj_projection.pth"
METADATA_2D_GLOBAL = "/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat4_2D_redo1_tunejoint1_trial5_proj/cat4_2D_redo1_tunejoint1_trial5_proj_prototype_metadata.json"
DIMENSION_2D_GLOBAL = "2D"
LABEL_SET_2D_GLOBAL = "4"
PROTO_TIME_LEN_2D_GLOBAL = 32
PROTOTYPES_PER_CLASS_2D_GLOBAL = 7
BACKBONE_2D_GLOBAL = "resnet18"
# shared
PROTO_DIM = 512
BATCH_SIZE = 32
TOP_K = 5 # number of top prototypes to select for prototype analysis
CUSTOM_GROUPS = True
JOINT_PROTOTYPES_PER_BORDER = 0

PTBXL_DATASET_PATH = "/gpfs/data/bbj-lab/users/sethis/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
#ALSO update the --output-path in the arguments in the main function below

def console_debug(label, message, pretty=True):
    if pretty:
        print(f"[DEBUG-{label}]")
        pprint.pprint(message)
    else:
        print(f"[DEBUG-{label}] ")
        print(f"{message}")

def load_fusion_model_weights(model, weights_path):
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    # PyTorch Lightning saves weights under 'state_dict'
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    # Remove 'model.' or 'net.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[len('model.'):]] = v
        elif k.startswith('net.'):
            new_state_dict[k[len('net.'):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights from {weights_path}")
    return model

def get_branch_dataloader(dimension, label_set):
    return get_dataloaders(
        batch_size=BATCH_SIZE, 
        mode=dimension,
        sampling_rate=100, 
        label_set=label_set,
        work_num=0,
        return_sample_ids=True,
        custom_groups=CUSTOM_GROUPS, 
        standardize=False,
        remove_baseline=True,
    )
   
def get_fusion_test_dataloader(target_ecg):    
    # Get regular fusion test dataloader if target_ecg is None. If target_ecg is specified, return a standard dataloader
    # toso that our samples contain ECG ids. This is a workaround since using return_sample_ids=True with 
    # the get_fusion_dataloaders function doesn't work.
    
    if target_ecg == None:
        # Create args for fusion dataloader
        class Args:
            def __init__(self):
                self.batch_size = BATCH_SIZE
                self.sampling_rate = 100
                self.standardize = False
                self.remove_baseline = True
                self.num_workers = 0
    
        fusion_dataloader_args = Args()
        _, _, test_loader_fusion, _ = get_fusion_dataloaders(fusion_dataloader_args, return_sample_ids=False)

        return test_loader_fusion   
    else:
        _, _, standard_test_loader, _ = get_dataloaders(
            batch_size=1,
            mode="2D",
            sampling_rate=100,
            standardize=False,
            remove_baseline=True,
            return_sample_ids=True,
            custom_groups=False,
            label_set="all",
            work_num=0
        )

        return standard_test_loader

def load_models(label_map_branches, label_map_fusion):
    num_classes1 = len(label_map_branches["1"])
    num_classes3 = len(label_map_branches["3"])
    num_classes4 = len(label_map_branches["4"])
    num_classes_fusion = len(label_map_fusion)
    console_debug('num_classes_fusion', num_classes_fusion) 

    # Load pretrained branch models
    model_1d = ProtoECGNet1D(
        num_classes=num_classes1,
        single_class_prototype_per_class=PROTOTYPES_PER_CLASS_1D,
        joint_prototypes_per_border=JOINT_PROTOTYPES_PER_BORDER,
        proto_dim=PROTO_DIM,
        backbone=BACKBONE_1D,
        prototype_activation_function='log',
        latent_space_type='arc',
        add_on_layers_type='linear',
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=CUSTOM_GROUPS,
        label_set=LABEL_SET_1D,
        pretrained_weights=WEIGHTS_1D
    )

    model_2d_partial = ProtoECGNet2D(
        num_classes=num_classes3,
        single_class_prototype_per_class=PROTOTYPES_PER_CLASS_2D_PARTIAL,
        joint_prototypes_per_border=JOINT_PROTOTYPES_PER_BORDER,
        proto_dim=PROTO_DIM,
        backbone=BACKBONE_2D_PARTIAL,
        prototype_activation_function='log',
        latent_space_type='arc',
        add_on_layers_type='linear',
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=CUSTOM_GROUPS,
        label_set=LABEL_SET_2D_PARTIAL,
        pretrained_weights=WEIGHTS_2D_PARTIAL,
        proto_time_len=PROTO_TIME_LEN_2D_PARTIAL
    )

    model_2d_global = ProtoECGNet2D(
        num_classes=num_classes4,
        single_class_prototype_per_class=PROTOTYPES_PER_CLASS_2D_GLOBAL,
        joint_prototypes_per_border=JOINT_PROTOTYPES_PER_BORDER,
        proto_dim=PROTO_DIM,
        backbone=BACKBONE_2D_GLOBAL,
        prototype_activation_function='log',
        latent_space_type='arc',
        add_on_layers_type='linear',
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=CUSTOM_GROUPS,
        label_set=LABEL_SET_2D_GLOBAL,
        pretrained_weights=WEIGHTS_2D_GLOBAL,
        proto_time_len=PROTO_TIME_LEN_2D_GLOBAL
    )

    # Load pre-trained fusion model
    fusion_model = FusionProtoClassifier(model_1d, model_2d_partial, model_2d_global, num_classes=num_classes_fusion)

    # Load fusion classifier weights if available
    if os.path.exists(FUSION_WEIGHTS):
        fusion_model = load_fusion_model_weights(fusion_model, FUSION_WEIGHTS)               
    else:
        print(f"Error loading pretrained weights for fusion classifier from: {FUSION_WEIGHTS}")

    return fusion_model, model_1d, model_2d_partial, model_2d_global

def load_metadata(path):
    # Load prototype metadata
    try:
        with open(path, 'r') as f:
            prototype_metadata = json.load(f)
        print(f"Loaded prototype metadata with {len(prototype_metadata)} prototypes")
        return prototype_metadata
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {path}")
        prototype_metadata = {}
        return prototype_metadata

def format_batch_for_fusion(X_batch, y_batch, ecg_ids):
    """
    Convert standard batch to fusion format while preserving ECG IDs.
    
    Args:
        X_batch: Input tensors (N, 12, 1000) or (N, 1, 12, 1000)
        y_batch: Label tensors (N, num_classes)
        ecg_ids: ECG ID tensors (N,)
    
    Returns:
        List of tuples: [(X_fusion, labels_dict, ecg_id), ...]
    """
    results = []
    for X, y, ecg_id in zip(X_batch, y_batch, ecg_ids):
        # Ensure X has correct shape for fusion (1, 1, 12, 1000)
        if X.dim() == 2:  # (12, 1000)
            X_fusion = X.unsqueeze(0).unsqueeze(0)  # (1, 1, 12, 1000)
        elif X.dim() == 3:  # (1, 12, 1000)
            X_fusion = X.unsqueeze(0)  # (1, 1, 12, 1000)
        else:  # Already (1, 1, 12, 1000)
            X_fusion = X.unsqueeze(0) if X.dim() == 3 else X
        
        # Create fusion format labels_dict
        labels_dict = {
            "1D": y.unsqueeze(0),
            "2D_partial": y.unsqueeze(0),
            "2D_global": y.unsqueeze(0),
            "full": y.unsqueeze(0)
        }
        
        results.append((X_fusion, labels_dict, ecg_id.item()))
    return results

def get_target_ecg(test_loader, target_ecg):
    for batch in test_loader:
        X_batch, y_batch, sample_ids = batch
        for x, y, ecg_id in zip(X_batch, y_batch, sample_ids):
            if int(ecg_id.item()) == target_ecg:
                return x, y, ecg_id.item()

    raise ValueError(f"ECG {target_ecg} not found in the test set.")    

def run_inference(X_sample):
    X_sample = X_sample.to(device)
    with torch.no_grad():
        # Get logits from fusion model
        logits = fusion_model(X_sample)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # Get similarity scores from individual branches
        x1d = X_sample.squeeze(1)  # Convert [N, 1, 12, 1000] → [N, 12, 1000] for 1D branch
        _, _, sim1d = fusion_model.model1d(x1d)
        _, _, sim2d_partial = fusion_model.model2d_partial(X_sample)
        _, _, sim2d_global = fusion_model.model2d_global(X_sample)
        
        # Concatenate similarity scores
        similarity_scores = torch.cat([sim1d, sim2d_partial, sim2d_global], dim=1).cpu().numpy().flatten()
        
        return probabilities, similarity_scores

def get_predicted_classes_indices(probabilities):
    return np.where(probabilities >= 0.5)[0]

def get_predicted_labels(label_map_fusion, probabilities):
    return list(zip(label_map_fusion, probabilities))

def get_true_labels_by_branch(true_label_names, label_map_branches):
    """
    Map true labels to their respective branches.
    
    Args:
        true_label_names: List of true label names for the sample
        label_map_branches: Dict containing branch-specific label lists
    
    Returns:
        dict: Branch-specific labels {'1D': [...], '2D_partial': [...], '2D_global': [...]}
    """
    branch_labels = {"1D": [], "2D_partial": [], "2D_global": []}
    
    # Map each true label to its branch
    for label_name in true_label_names:
        if label_name in label_map_branches["1"]:
            branch_labels["1D"].append(label_name)
        elif label_name in label_map_branches["3"]:
            branch_labels["2D_partial"].append(label_name)
        elif label_name in label_map_branches["4"]:
            branch_labels["2D_global"].append(label_name)
    
    return branch_labels

def print_overall_predictions(probabilities, label_map_fusion, y_sample, label_map_branches):
    print("\n--- Fusion Model Predictions (Threshold: 0.5) ---")        
    predicted_classes_indices = get_predicted_classes_indices(probabilities)
    predicted_labels = get_predicted_labels(label_map_fusion, probabilities)

    # Display predictions with correct label names
    if len(predicted_classes_indices) > 0:
        # Sort by probability (highest first)
        sorted_predictions = sorted(predicted_labels, key=lambda x: x[1], reverse=True)
        for label, prob in sorted_predictions:
            if prob >= 0.5:
                print(f"  {label}: {prob:.4f}")
    else:
        print("  No classes predicted above threshold (0.5)")

    # Get true labels from y_sample
    true_labels_indices = np.where(y_sample.cpu().numpy() == 1)[0]
    true_label_names = [label_map_fusion[i] for i in true_labels_indices]

    print("\n--- True Labels ---")
    if true_label_names:
        print(f"  {', '.join(true_label_names)}")
    else:
        print("  No true labels for this sample.")

    true_labels_by_branch = get_true_labels_by_branch(true_label_names, label_map_branches)
    print("\n--- True Labels by Branch ---")
    if true_labels_by_branch["1D"]:
        print(f"  1D Rhythm: {', '.join(true_labels_by_branch['1D'])}")
    if true_labels_by_branch["2D_partial"]:
        print(f"  2D Partial: {', '.join(true_labels_by_branch['2D_partial'])}")
    if true_labels_by_branch["2D_global"]:
        print(f"  2D Global: {', '.join(true_labels_by_branch['2D_global'])}")

def print_top_protos_by_branch(similarity_scores, num_prototypes_1d, num_prototypes_2d_partial, num_prototypes_2d_global, 
                              prototype_metadata1, prototype_metadata3, prototype_metadata4, 
                              label_map_branches, top_k=5):
    """
    Print top activated prototypes for each branch separately.
    
    Args:
        similarity_scores: Full similarity scores from fusion model
        num_prototypes_1d: Number of 1D prototypes
        num_prototypes_2d_partial: Number of 2D partial prototypes  
        num_prototypes_2d_global: Number of 2D global prototypes
        prototype_metadata1: Metadata for 1D prototypes
        prototype_metadata3: Metadata for 2D partial prototypes
        prototype_metadata4: Metadata for 2D global prototypes
        label_map_branches: Dictionary with branch-specific label mappings
        top_k: Number of top prototypes to show per branch
    """
    print("\n--- Top Prototypes by Branch ---")

    # Split similarity scores by branch
    sim1d = similarity_scores[:num_prototypes_1d]
    sim2d_partial = similarity_scores[num_prototypes_1d:num_prototypes_1d + num_prototypes_2d_partial]
    sim2d_global = similarity_scores[num_prototypes_1d + num_prototypes_2d_partial:]
    
    # Get top prototypes for each branch
    top_1d_indices = np.argsort(sim1d)[-top_k:][::-1]  # Highest first
    top_2d_partial_indices = np.argsort(sim2d_partial)[-top_k:][::-1]
    top_2d_global_indices = np.argsort(sim2d_global)[-top_k:][::-1]
    
    # Print 1D Rhythm prototypes
    print(f"\n  1D Rhythm (Top {top_k}):")
    for i, proto_idx in enumerate(top_1d_indices):
        similarity = sim1d[proto_idx]
        
        # Get prototype metadata
        proto_id_str = str(proto_idx)
        if proto_id_str in prototype_metadata1:
            meta = prototype_metadata1[proto_id_str]
            assigned_class = meta.get('prototype_class', 'N/A')
            ecg_id = meta.get('ecg_id', 'N/A')
            
            # Get branch-specific true labels
            true_labels = meta.get('true_labels', [])
            branch_true_labels = []
            for j, val in enumerate(true_labels):
                if j < len(label_map_branches["1"]) and val == 1.0:
                    branch_true_labels.append(label_map_branches["1"][j])
            
            # Get complete true labels
            complete_true_labels = get_complete_true_labels_from_ptbxl(ecg_id)
            
            print(f"    {i+1}. Prototype {proto_idx} ({assigned_class}) - Similarity {similarity:.4f}")
            print(f"        ECG ID: {ecg_id}")
            if branch_true_labels:
                print(f"        Branch True Labels: {', '.join(branch_true_labels)}")
            else:
                print(f"        Branch True Labels: None")
            if complete_true_labels:
                print(f"        Complete True Labels: {', '.join(complete_true_labels)}")
            else:
                print(f"        Complete True Labels: None")
        else:
            print(f"    {i+1}. Prototype {proto_idx} - Similarity {similarity:.4f}")
    
    # Print 2D Partial prototypes  
    print(f"\n  2D Partial (Top {top_k}):")
    for i, proto_idx in enumerate(top_2d_partial_indices):
        similarity = sim2d_partial[proto_idx]
        
        # Get prototype metadata
        proto_id_str = str(proto_idx)
        if proto_id_str in prototype_metadata3:
            meta = prototype_metadata3[proto_id_str]
            assigned_class = meta.get('prototype_class', 'N/A')
            ecg_id = meta.get('ecg_id', 'N/A')
            
            # Get branch-specific true labels
            true_labels = meta.get('true_labels', [])
            branch_true_labels = []
            for j, val in enumerate(true_labels):
                if j < len(label_map_branches["3"]) and val == 1.0:
                    branch_true_labels.append(label_map_branches["3"][j])
            
            # Get complete true labels
            complete_true_labels = get_complete_true_labels_from_ptbxl(ecg_id)
            
            print(f"    {i+1}. Prototype {proto_idx} ({assigned_class}) - Similarity {similarity:.4f}")
            print(f"        ECG ID: {ecg_id}")
            if branch_true_labels:
                print(f"        Branch True Labels: {', '.join(branch_true_labels)}")
            else:
                print(f"        Branch True Labels: None")
            if complete_true_labels:
                print(f"        Complete True Labels: {', '.join(complete_true_labels)}")
            else:
                print(f"        Complete True Labels: None")
        else:
            print(f"    {i+1}. Prototype {proto_idx} - Similarity {similarity:.4f}")
    
    # Print 2D Global prototypes
    print(f"\n  2D Global (Top {top_k}):")
    for i, proto_idx in enumerate(top_2d_global_indices):
        similarity = sim2d_global[proto_idx]
        
        # Get prototype metadata
        proto_id_str = str(proto_idx)
        if proto_id_str in prototype_metadata4:
            meta = prototype_metadata4[proto_id_str]
            assigned_class = meta.get('prototype_class', 'N/A')
            ecg_id = meta.get('ecg_id', 'N/A')
            
            # Get branch-specific true labels
            true_labels = meta.get('true_labels', [])
            branch_true_labels = []
            for j, val in enumerate(true_labels):
                if j < len(label_map_branches["4"]) and val == 1.0:
                    branch_true_labels.append(label_map_branches["4"][j])
            
            # Get complete true labels
            complete_true_labels = get_complete_true_labels_from_ptbxl(ecg_id)
            
            print(f"    {i+1}. Prototype {proto_idx} ({assigned_class}) - Similarity {similarity:.4f}")
            print(f"        ECG ID: {ecg_id}")
            if branch_true_labels:
                print(f"        Branch True Labels: {', '.join(branch_true_labels)}")
            else:
                print(f"        Branch True Labels: None")
            if complete_true_labels:
                print(f"        Complete True Labels: {', '.join(complete_true_labels)}")
            else:
                print(f"        Complete True Labels: None")
        else:
            print(f"    {i+1}. Prototype {proto_idx} - Similarity {similarity:.4f}")

    return

def get_prototype_class_mappings(prototype_metadata1, prototype_metadata3, prototype_metadata4):
    """
    Create prototype to class mappings for each branch.
    Returns: dict with branch keys and prototype_id -> class_name mappings
    """
    return {
        "1D": {int(proto_id): meta["prototype_class"] for proto_id, meta in prototype_metadata1.items()},
        "2D_partial": {int(proto_id): meta["prototype_class"] for proto_id, meta in prototype_metadata3.items()},
        "2D_global": {int(proto_id): meta["prototype_class"] for proto_id, meta in prototype_metadata4.items()}
    }

def get_complete_true_labels_from_ptbxl(ecg_id):
    """
    Get complete true labels from PTB-XL database for a given ECG ID.
    """
    try:
        df = pd.read_csv(PTBXL_DATASET_PATH, index_col="ecg_id")
        if ecg_id in df.index:
            scp_codes = ast.literal_eval(df.loc[ecg_id, 'scp_codes'])
            return list(scp_codes.keys())
        else:
            return []
    except Exception as e:
        print(f"Error loading complete true labels for ECG {ecg_id}: {e}")
        return []

def get_prototype_metadata_by_class(prototype_metadata1, prototype_metadata3, prototype_metadata4, 
                                   similarity_scores, num_prototypes_1d, num_prototypes_2d_partial, 
                                   num_prototypes_2d_global, class_name):
    """
    Get all prototypes assigned to a specific class with their metadata.
    Returns: list of prototype info dictionaries
    """
    sim1d = similarity_scores[:num_prototypes_1d]
    sim2d_partial = similarity_scores[num_prototypes_1d:num_prototypes_1d + num_prototypes_2d_partial]
    sim2d_global = similarity_scores[num_prototypes_1d + num_prototypes_2d_partial:]
    
    class_prototypes = []
    
    # 1D branch prototypes
    for proto_id_str, meta in prototype_metadata1.items():
        proto_idx = int(proto_id_str)
        if proto_idx < len(sim1d) and meta.get('prototype_class') == class_name:
            # Get complete true labels from PTB-XL database
            ecg_id = meta.get('ecg_id', 'N/A')
            complete_true_labels = get_complete_true_labels_from_ptbxl(ecg_id)
            
            class_prototypes.append({
                'branch': '1D Rhythm',
                'proto_id': proto_idx,
                'assigned_class': meta.get('prototype_class', 'N/A'),
                'raw_similarity': sim1d[proto_idx],
                'true_labels': meta.get('true_labels', []),
                'ecg_id': ecg_id,
                'complete_true_labels': complete_true_labels
            })
    
    # 2D Partial branch prototypes
    for proto_id_str, meta in prototype_metadata3.items():
        proto_idx = int(proto_id_str)
        if proto_idx < len(sim2d_partial) and meta.get('prototype_class') == class_name:
            # Get complete true labels from PTB-XL database
            ecg_id = meta.get('ecg_id', 'N/A')
            complete_true_labels = get_complete_true_labels_from_ptbxl(ecg_id)
            
            class_prototypes.append({
                'branch': '2D Partial',
                'proto_id': proto_idx,
                'assigned_class': meta.get('prototype_class', 'N/A'),
                'raw_similarity': sim2d_partial[proto_idx],
                'true_labels': meta.get('true_labels', []),
                'ecg_id': ecg_id,
                'complete_true_labels': complete_true_labels
            })
    
    # 2D Global branch prototypes
    for proto_id_str, meta in prototype_metadata4.items():
        proto_idx = int(proto_id_str)
        if proto_idx < len(sim2d_global) and meta.get('prototype_class') == class_name:
            # Get complete true labels from PTB-XL database
            ecg_id = meta.get('ecg_id', 'N/A')
            complete_true_labels = get_complete_true_labels_from_ptbxl(ecg_id)
            
            class_prototypes.append({
                'branch': '2D Global',
                'proto_id': proto_idx,
                'assigned_class': meta.get('prototype_class', 'N/A'),
                'raw_similarity': sim2d_global[proto_idx],
                'true_labels': meta.get('true_labels', []),
                'ecg_id': ecg_id,
                'complete_true_labels': complete_true_labels
            })
    
    return class_prototypes

def format_prototype_display_info(proto_info, label_map_branches):
    """
    Format prototype information for display.
    Returns: dict with formatted strings for display
    """
    # Get branch-specific true labels
    branch_true_labels = []
    if proto_info['branch'] == '1D Rhythm':
        for i, val in enumerate(proto_info['true_labels']):
            if i < len(label_map_branches["1"]) and val == 1.0:
                branch_true_labels.append(label_map_branches["1"][i])
    elif proto_info['branch'] == '2D Partial':
        for i, val in enumerate(proto_info['true_labels']):
            if i < len(label_map_branches["3"]) and val == 1.0:
                branch_true_labels.append(label_map_branches["3"][i])
    elif proto_info['branch'] == '2D Global':
        for i, val in enumerate(proto_info['true_labels']):
            if i < len(label_map_branches["4"]) and val == 1.0:
                branch_true_labels.append(label_map_branches["4"][i])
    
    return {
        'branch': proto_info['branch'],
        'proto_id': proto_info['proto_id'],
        'assigned_class': proto_info['assigned_class'],
        'raw_similarity': proto_info['raw_similarity'],
        'ecg_id': proto_info['ecg_id'],
        'branch_true_labels': branch_true_labels,
        'complete_true_labels': proto_info.get('complete_true_labels', [])
    }

def print_top_prototypes_by_class(probabilities, similarity_scores, label_map_fusion, 
                                 prototype_metadata1, prototype_metadata3, prototype_metadata4,
                                 model_1d, model_2d_partial, model_2d_global, 
                                 label_map_branches, threshold=0.5, top_n=5):
    """
    Print top activated prototypes for each predicted class.
    Shows prototypes assigned to each predicted class, sorted by similarity.
    """
    predicted_classes_indices = np.where(probabilities >= threshold)[0]
    if len(predicted_classes_indices) == 0:
        print("No classes predicted above the threshold.")
        return
    
    # Get branch prototype counts
    num_prototypes_1d = model_1d.num_prototypes
    num_prototypes_2d_partial = model_2d_partial.num_prototypes
    num_prototypes_2d_global = model_2d_global.num_prototypes
    
    for class_idx in predicted_classes_indices:
        class_name = label_map_fusion[class_idx]
        print(f"\n=== Top {top_n} Most Activated Prototypes for {class_name} ===")
        
        # Get prototypes assigned to this class
        class_prototypes = get_prototype_metadata_by_class(
            prototype_metadata1, prototype_metadata3, prototype_metadata4,
            similarity_scores, num_prototypes_1d, num_prototypes_2d_partial, num_prototypes_2d_global,
            class_name
        )
        
        if not class_prototypes:
            print(f"  No prototypes assigned to {class_name} found.")
            continue
        
        # Sort by similarity and show top prototypes
        class_prototypes.sort(key=lambda x: x['raw_similarity'], reverse=True)
        
        print(f"  Top {min(top_n, len(class_prototypes))} most activated {class_name} prototypes:")
        for i, proto_info in enumerate(class_prototypes[:top_n]):
            formatted_info = format_prototype_display_info(proto_info, label_map_branches)
            
            print(f"    {i+1}. {formatted_info['branch']} Prototype {formatted_info['proto_id']}")
            print(f"        Assigned Class: {formatted_info['assigned_class']}")
            print(f"        Raw Similarity: {formatted_info['raw_similarity']:.4f}")
            print(f"        ECG ID: {formatted_info['ecg_id']}")
            
            if formatted_info['branch_true_labels']:
                print(f"        Branch True Labels: {', '.join(formatted_info['branch_true_labels'])}")
            else:
                print(f"        Branch True Labels: None")
            
            if formatted_info['complete_true_labels']:
                print(f"        Complete True Labels: {', '.join(formatted_info['complete_true_labels'])}")
            else:
                print(f"        Complete True Labels: None")
            print()

def save_results(ecg_id, probabilities, similarity_scores, label_map_fusion, y_sample, label_map_branches,
                prototype_metadata1, prototype_metadata3, prototype_metadata4,
                model_1d, model_2d_partial, model_2d_global, output_path):
    """
    Save inference results to CSV file.
    
    Args:
        ecg_id: ECG ID of the sample
        probabilities: Fusion model probabilities
        similarity_scores: Similarity scores from all branches
        label_map_fusion: Fusion model label mapping
        y_sample: True labels for the sample
        label_map_branches: Branch-specific label mappings
        prototype_metadata1/3/4: Prototype metadata for each branch
        model_1d/2d_partial/2d_global: Branch models
        output_path: Path to save CSV file
    """
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    # Get fusion predictions using existing function
    predicted_classes_indices = get_predicted_classes_indices(probabilities)
    fusion_predictions = {}
    for i in predicted_classes_indices:
        fusion_predictions[label_map_fusion[i]] = float(probabilities[i])
    
    # Get true labels
    true_labels_indices = np.where(y_sample.cpu().numpy() == 1)[0]
    true_labels = [label_map_fusion[i] for i in true_labels_indices]
    
    # Get branch-specific true labels
    true_labels_by_branch = get_true_labels_by_branch(true_labels, label_map_branches)
    
    # Get top prototypes by branch
    def get_top_protos_by_branch(similarity_scores, num_prototypes, prototype_metadata, branch_label_map, top_k=5):
        """Helper function to get top prototypes for a branch"""
        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        top_protos = []
        
        for proto_idx in top_indices:
            proto_id_str = str(proto_idx)
            if proto_id_str in prototype_metadata:
                meta = prototype_metadata[proto_id_str]
                
                # Get branch-specific true labels
                true_labels = meta.get('true_labels', [])
                branch_true_labels = []
                for j, val in enumerate(true_labels):
                    if j < len(branch_label_map) and val == 1.0:
                        branch_true_labels.append(branch_label_map[j])
                
                # Get complete true labels
                ecg_id = meta.get('ecg_id', 'N/A')
                complete_true_labels = get_complete_true_labels_from_ptbxl(ecg_id)
                
                proto_info = {
                    "prototype_num": proto_idx,
                    "prototype_class": meta.get('prototype_class', 'N/A'),
                    "similarity": float(similarity_scores[proto_idx]),
                    "ecg_id": ecg_id,
                    "all_true_labels": complete_true_labels,
                    "branch_true_labels": branch_true_labels
                }
                top_protos.append(proto_info)
            else:
                # Fallback if metadata not found
                proto_info = {
                    "prototype_num": proto_idx,
                    "prototype_class": "N/A",
                    "similarity": float(similarity_scores[proto_idx]),
                    "ecg_id": "N/A",
                    "all_true_labels": [],
                    "branch_true_labels": []
                }
                top_protos.append(proto_info)
        
        return top_protos
    
    # Get top prototypes for each branch
    num_prototypes_1d = model_1d.num_prototypes
    num_prototypes_2d_partial = model_2d_partial.num_prototypes
    num_prototypes_2d_global = model_2d_global.num_prototypes
    
    sim1d = similarity_scores[:num_prototypes_1d]
    sim2d_partial = similarity_scores[num_prototypes_1d:num_prototypes_1d + num_prototypes_2d_partial]
    sim2d_global = similarity_scores[num_prototypes_1d + num_prototypes_2d_partial:]
    
    top_protos_1d = get_top_protos_by_branch(sim1d, num_prototypes_1d, prototype_metadata1, label_map_branches["1"])
    top_protos_2d_partial = get_top_protos_by_branch(sim2d_partial, num_prototypes_2d_partial, prototype_metadata3, label_map_branches["3"])
    top_protos_2d_global = get_top_protos_by_branch(sim2d_global, num_prototypes_2d_global, prototype_metadata4, label_map_branches["4"])
    
    # Get top prototypes by class
    top_protos_by_class = []
    predicted_classes_indices = np.where(probabilities >= 0.5)[0]
    
    for class_idx in predicted_classes_indices:
        class_name = label_map_fusion[class_idx]
        
        # Get prototypes assigned to this class
        class_prototypes = get_prototype_metadata_by_class(
            prototype_metadata1, prototype_metadata3, prototype_metadata4,
            similarity_scores, num_prototypes_1d, num_prototypes_2d_partial, num_prototypes_2d_global,
            class_name
        )
        
        if class_prototypes:
            # Sort by similarity and get top 5
            class_prototypes.sort(key=lambda x: x['raw_similarity'], reverse=True)
            top_class_protos = []
            
            for proto_info in class_prototypes[:5]:
                # Determine branch group
                if proto_info['branch'] == '1D Rhythm':
                    group = "1"
                elif proto_info['branch'] == '2D Partial':
                    group = "3"
                elif proto_info['branch'] == '2D Global':
                    group = "4"
                else:
                    group = "unknown"
                
                # Get complete true labels
                complete_true_labels = get_complete_true_labels_from_ptbxl(proto_info['ecg_id'])
                
                # Get branch-specific true labels
                branch_true_labels = []
                if proto_info['branch'] == '1D Rhythm':
                    for i, val in enumerate(proto_info['true_labels']):
                        if i < len(label_map_branches["1"]) and val == 1.0:
                            branch_true_labels.append(label_map_branches["1"][i])
                elif proto_info['branch'] == '2D Partial':
                    for i, val in enumerate(proto_info['true_labels']):
                        if i < len(label_map_branches["3"]) and val == 1.0:
                            branch_true_labels.append(label_map_branches["3"][i])
                elif proto_info['branch'] == '2D Global':
                    for i, val in enumerate(proto_info['true_labels']):
                        if i < len(label_map_branches["4"]) and val == 1.0:
                            branch_true_labels.append(label_map_branches["4"][i])
                
                proto_data = {
                    "prototype_num": proto_info['proto_id'],
                    "group": group,
                    "prototype_class": proto_info['assigned_class'],
                    "similarity": float(proto_info['raw_similarity']),
                    "ecg_id": proto_info['ecg_id'],
                    "all_true_labels": complete_true_labels,
                    "branch_true_labels": branch_true_labels
                }
                top_class_protos.append(proto_data)
            
            class_data = {
                "predicted_class": class_name,
                "top_protos": top_class_protos
            }
            top_protos_by_class.append(class_data)
    
    # Create result row
    result_row = {
        "ecg_id": convert_numpy_types(ecg_id),
        "fusion_predictions": json.dumps(convert_numpy_types(fusion_predictions)),
        "true_labels": json.dumps(convert_numpy_types(true_labels)),
        "1d_true_labels": json.dumps(convert_numpy_types(true_labels_by_branch["1D"])),
        "2d_partial_true_labels": json.dumps(convert_numpy_types(true_labels_by_branch["2D_partial"])),
        "2d_global_true_labels": json.dumps(convert_numpy_types(true_labels_by_branch["2D_global"])),
        "top_protos_1d": json.dumps(convert_numpy_types(top_protos_1d)),
        "top_protos_2d_partial": json.dumps(convert_numpy_types(top_protos_2d_partial)),
        "top_protos_2d_global": json.dumps(convert_numpy_types(top_protos_2d_global)),
        "top_protos_by_class": json.dumps(convert_numpy_types(top_protos_by_class))
    }
    
    # Save to CSV
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(output_path)
    
    df = pd.DataFrame([result_row])
    
    if file_exists:
        # Append without headers
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # Write with headers
        df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fusion model inference and prototype analysis.")
    parser.add_argument('--target-ecg', type=int, default=None, help='Target specific ECG ID for analysis')
    parser.add_argument('--num-samples', type=int, default=-1, help='Number of samples to process (default: 5, use -1 for all)')
    parser.add_argument('--output-path', type=str, default="/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/fusion_inference_script_results/script_results.csv", help='Path to output results CSV to')

    args = parser.parse_args()
    target_ecg = args.target_ecg

    seed_everything(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    # Get test dataloaders
    _, _, test_loader_1d, _ = get_branch_dataloader(DIMENSION_1D, LABEL_SET_1D)
    _, _, test_loader_2d_partial, _ = get_branch_dataloader(DIMENSION_2D_PARTIAL, LABEL_SET_2D_PARTIAL)
    _, _, test_loader_2d_global, _ = get_branch_dataloader(DIMENSION_2D_GLOBAL, LABEL_SET_2D_GLOBAL)

    # Get fusion dataloader
    # Note that when running inference on a single target ecg sample we need to format the sample for fusion inference
    test_loader_fusion = get_fusion_test_dataloader(target_ecg)  

    # Load label mappings    
    original_label_mappings = load_label_mappings(custom_groups=False)
    label_names = original_label_mappings["all"] # Uses the ordering from the original ungrouped scp_statements.csv

    # This gives the correct label maps for the individual branch models, but the wrong label map for the fusion model
    label_map_branches = load_fusion_label_mappings()

    # Get the correctly ordered labels for the fusion model
    label_map_fusion = load_label_mappings(custom_groups=False)["all"]

    # Load models
    fusion_model, model_1d, model_2d_partial, model_2d_global = load_models(label_map_branches, label_map_fusion)   

    fusion_model = fusion_model.to(device)
    fusion_model.eval()

    # Load prototype metadata for each branch
    prototype_metadata1 = load_metadata(METADATA_1D)
    prototype_metadata3 = load_metadata(METADATA_2D_PARTIAL)
    prototype_metadata4 = load_metadata(METADATA_2D_GLOBAL)

    if target_ecg is not None:
        # For single target samples, we need to format the sample for fusion inference
        X_sample, y_sample, ecg_id = get_target_ecg(test_loader_fusion, target_ecg)
        
        # Use helper function to format for fusion
        formatted_samples = format_batch_for_fusion(
            X_sample,  # format_batch_for_fusion handles unsqueeze
            y_sample,  # format_batch_for_fusion handles unsqueeze
            torch.tensor([ecg_id])  # Create tensor for ECG ID
        )
        X_fusion, labels_dict, ecg_id = formatted_samples[0]
        
        # Run inference
        print(f"\n" + "="*50)
        print(f"STARTING INFERENCE FOR ECG {ecg_id}")
        print("="*50)

        probabilities, similarity_scores = run_inference(X_fusion)
        print_overall_predictions(probabilities, label_map_fusion, y_sample, label_map_branches)

        print_top_protos_by_branch(
            similarity_scores, 
            model_1d.num_prototypes, 
            model_2d_partial.num_prototypes, 
            model_2d_global.num_prototypes,
            prototype_metadata1,
            prototype_metadata3,
            prototype_metadata4,
            label_map_branches,
            top_k=5
        )

        print_top_prototypes_by_class(
            probabilities, 
            similarity_scores, 
            label_map_fusion, 
            prototype_metadata1, 
            prototype_metadata3, 
            prototype_metadata4,
            model_1d, 
            model_2d_partial, 
            model_2d_global, 
            label_map_branches,
            threshold=0.5, 
            top_n=5
        )

        # Save results
        save_results(
            ecg_id, probabilities, similarity_scores, label_map_fusion, y_sample, label_map_branches,
            prototype_metadata1, prototype_metadata3, prototype_metadata4,
            model_1d, model_2d_partial, model_2d_global, args.output_path
        )
    else:
        # Get samples from standard dataloader to preserve ECG IDs
        _, _, test_loader_standard, _ = get_dataloaders(
            batch_size=1,
            mode="2D",
            sampling_rate=100,
            standardize=False,
            remove_baseline=True,
            return_sample_ids=True,
            custom_groups=False,
            label_set="all",
            work_num=0
        )
        
        test_iter = iter(test_loader_standard)
        num_samples = args.num_samples if args.num_samples > 0 else float('inf')
        sample_idx = 0
        
        print(f"Starting inference loop: num_samples={num_samples}")
        
        while sample_idx < num_samples:
            try:
                batch = next(test_iter)
                
                # Standard dataloader returns (inputs, labels, ecg_ids)
                X_batch, y_batch, ecg_ids = batch
                
                # Use helper function to format for fusion
                formatted_samples = format_batch_for_fusion(X_batch, y_batch, ecg_ids)
                X_fusion, labels_dict, ecg_id = formatted_samples[0]
            
                print(f"\n" + "="*50)
                print(f"STARTING INFERENCE FOR SAMPLE {sample_idx} (ECG ID {ecg_id})")
                print("="*50)
                X_fusion = X_fusion.to(device)
                
                # Get true labels from labels_dict
                y_sample = labels_dict["full"][0]  # Take first sample's labels
                
                # Run inference
                probabilities, similarity_scores = run_inference(X_fusion)
                
                # Display results
                print_overall_predictions(probabilities, label_map_fusion, y_sample, label_map_branches)

                print_top_protos_by_branch(
                    similarity_scores, 
                    model_1d.num_prototypes, 
                    model_2d_partial.num_prototypes, 
                    model_2d_global.num_prototypes,
                    prototype_metadata1,
                    prototype_metadata3,
                    prototype_metadata4,
                    label_map_branches,
                    top_k=5
                )

                print_top_prototypes_by_class(
                    probabilities, 
                    similarity_scores, 
                    label_map_fusion, 
                    prototype_metadata1, 
                    prototype_metadata3, 
                    prototype_metadata4,
                    model_1d, 
                    model_2d_partial, 
                    model_2d_global, 
                    label_map_branches,
                    threshold=0.5, 
                    top_n=5
                )

                # Save results for this sample
                save_results(
                    ecg_id, probabilities, similarity_scores, label_map_fusion, y_sample, label_map_branches,
                    prototype_metadata1, prototype_metadata3, prototype_metadata4,
                    model_1d, model_2d_partial, model_2d_global, args.output_path
                )

                print("\n" + "="*50)
                print(f"COMPLETED INFERENCE FOR SAMPLE {sample_idx} (ECG ID {ecg_id})")
                print("="*50)
                    
            except StopIteration:
                print(f"\n" + "="*50)
                print(f"REACHED END OF TEST SET - PROCESSED {sample_idx} SAMPLES")
                print("="*50)
                break
            except Exception as e:
                print(f"\n" + "="*50)
                print(f"ERROR PROCESSING SAMPLE {sample_idx} (ECG ID {ecg_id}): {e}")
                print("="*50)
                sample_idx += 1
                continue
            
            sample_idx += 1

    print("\n" + "="*50)
    print("END SCRIPT")
    print("="*50)


        

    
    


