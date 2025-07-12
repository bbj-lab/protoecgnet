"""
ProtoECGNet Fusion Inference Script

This script provides standalone inference capabilities for trained ProtoECGNet fusion models
that combine 1D rhythm, 2D partial morphology, and 2D global branches.

Usage:
    python inference_fusion.py --show-prototypes --sanity-check --save-results
    python inference_fusion.py --check-labels --show-metadata --save-results
"""

import torch
import numpy as np
import json
import random
import os
import argparse
import pandas as pd
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D
from proto_models2D import ProtoECGNet2D
from fusion import FusionProtoClassifier, load_fusion_label_mappings, get_fusion_dataloaders
from training_functions import seed_everything
from sklearn.metrics import roc_auc_score, f1_score

# Fusion Model Configuration
FUSION_WEIGHTS = "/path/to/fusion_classifier_contrastive_weights.ckpt"

# Individual branch weights (for loading the three sub-models)
# these can be the projected ones that you used to train the fusion classifier
FUSION_WEIGHTS1 = "/path/to/cat1_1D_branch_weights.pth" 
FUSION_WEIGHTS3 = "/path/to/cat3_2Dpartial_branch_weights.pth"
FUSION_WEIGHTS4 = "/path/to/cat4_2Dglobal_branch_weights.pth"

# Individual branch metadata files
METADATA_JSON1 = "/path/to/cat1_1D_proj_prototype_metadata"
METADATA_JSON3 = "/path/to/cat3_2Dpartial_proj_prototype_metadata"
METADATA_JSON4 = "/path/to/cat4_2Dglobal_proj_prototype_metadata"

# Model configurations for each branch
FUSION_BACKBONE1 = "resnet1d18"
FUSION_BACKBONE3 = "resnet18"
FUSION_BACKBONE4 = "resnet18"

FUSION_PROTO_DIM1 = 512
FUSION_PROTO_DIM3 = 512
FUSION_PROTO_DIM4 = 512

FUSION_SINGLE_PPC1 = 5
FUSION_SINGLE_PPC3 = 18
FUSION_SINGLE_PPC4 = 7

FUSION_JOINT_PPB1 = 0
FUSION_JOINT_PPB3 = 0
FUSION_JOINT_PPB4 = 0

# --- Utility Functions ---
def list_test_sample_ids(test_loader, max_samples=20):
    """
    Print the available test sample IDs (up to max_samples for brevity).
    """
    print("\n--- Available Test Sample IDs (showing up to {}): ---".format(max_samples))
    ids = []
    count = 0
    for batch in test_loader:
        if len(batch) == 3:  # Standard format
            _, _, id_batch = batch
        else:  # Fusion format
            _, labels_dict = batch
            # For fusion, we need to get IDs differently
            continue
        for ecg_id in id_batch:
            ids.append(int(ecg_id.item()))
            count += 1
            if count >= max_samples:
                break
        if count >= max_samples:
            break
    print(ids)
    if count == max_samples:
        print("... (truncated, more samples available)")
    return ids

def get_test_sample_by_id(test_loader, target_id):
    for batch in test_loader:
        if len(batch) == 3:  # Standard format
            X_batch, y_batch, id_batch = batch
            for x, y, ecg_id in zip(X_batch, y_batch, id_batch):
                if int(ecg_id.item()) == target_id:
                    return x, y, ecg_id.item()
        else:  # Fusion format
            X_batch, labels_dict = batch
            # For fusion, we need to handle this differently
            continue
    raise ValueError(f"ECG ID {target_id} not found in the test set.")

def get_random_test_sample(test_loader):
    test_iter = iter(test_loader)
    batch = next(test_iter)
    if len(batch) == 3:  # Standard format
        X_batch, y_batch, sample_ids = batch
        idx = random.randint(0, len(X_batch) - 1)
        return X_batch[idx], y_batch[idx], sample_ids[idx]
    else:  # Fusion format
        X_batch, labels_dict = batch
        idx = random.randint(0, len(X_batch) - 1)
        return X_batch[idx], labels_dict, idx

def load_model_weights(model, weights_path):
    if weights_path.endswith('.pth'):
        state_dict = torch.load(weights_path, map_location='cpu')
        # If saved with DataParallel, keys may have 'module.' prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path} (pth format)")
    elif weights_path.endswith('.ckpt'):
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
        print(f"Loaded weights from {weights_path} (ckpt format)")
    else:
        raise ValueError(f"Unsupported weights file format: {weights_path}")
    return model

def load_fusion_model():
    """
    Load the complete fusion model with all three sub-models.
    """
    print("Loading fusion model components...")
    
    # Load label mappings for each branch
    label_map = load_fusion_label_mappings()
    num_classes1 = len(label_map["1"])
    num_classes3 = len(label_map["3"])
    num_classes4 = len(label_map["4"])
    
    print(f"Creating category 1 model (1D rhythm) with backbone {FUSION_BACKBONE1}...")
    model_1d = ProtoECGNet1D(
        num_classes=num_classes1,
        single_class_prototype_per_class=FUSION_SINGLE_PPC1,
        joint_prototypes_per_border=FUSION_JOINT_PPB1,
        proto_dim=FUSION_PROTO_DIM1,
        backbone=FUSION_BACKBONE1,
        prototype_activation_function='arc',
        latent_space_type='arc',
        add_on_layers_type='linear',
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=True,
        label_set="1",
        pretrained_weights=FUSION_WEIGHTS1
    )
    
    print(f"Creating category 3 model (2D partial morphology) with backbone {FUSION_BACKBONE3}...")
    model_2d_partial = ProtoECGNet2D(
        num_classes=num_classes3,
        single_class_prototype_per_class=FUSION_SINGLE_PPC3,
        joint_prototypes_per_border=FUSION_JOINT_PPB3,
        proto_dim=FUSION_PROTO_DIM3,
        proto_time_len=3,
        backbone=FUSION_BACKBONE3,
        prototype_activation_function='arc',
        latent_space_type='arc',
        add_on_layers_type='linear',
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=True,
        label_set="3",
        pretrained_weights=FUSION_WEIGHTS3
    )
    
    print(f"Creating category 4 model (2D global) with backbone {FUSION_BACKBONE4}...")
    model_2d_global = ProtoECGNet2D(
        num_classes=num_classes4,
        single_class_prototype_per_class=FUSION_SINGLE_PPC4,
        joint_prototypes_per_border=FUSION_JOINT_PPB4,
        proto_dim=FUSION_PROTO_DIM4,
        proto_time_len=32,
        backbone=FUSION_BACKBONE4,
        prototype_activation_function='arc',
        latent_space_type='arc',
        add_on_layers_type='linear',
        class_specific=True,
        last_layer_connection_weight=1.0,
        m=0.05,
        dropout=0,
        custom_groups=True,
        label_set="4",
        pretrained_weights=FUSION_WEIGHTS4
    )
    
    # Create fusion model
    print("Creating fusion classifier...")
    fusion_model = FusionProtoClassifier(model_1d, model_2d_partial, model_2d_global, num_classes=71)
    
    # Load fusion classifier weights if available
    if os.path.exists(FUSION_WEIGHTS):
        fusion_model = load_model_weights(fusion_model, FUSION_WEIGHTS)
        print("Loaded fusion classifier weights")
    else:
        print("Warning: Fusion classifier weights not found, using untrained classifier")
    
    return fusion_model, model_1d, model_2d_partial, model_2d_global

def display_fusion_prototypes(probabilities, similarity_scores, label_names, 
                             prototype_metadata1, prototype_metadata3, prototype_metadata4,
                             model_1d, model_2d_partial, model_2d_global, threshold=0.5, top_n=5):
    """
    Display prototype analysis for fusion model, showing contributions from each branch.
    """
    print("\n--- Fusion Prototype Analysis ---")
    predicted_classes_indices = np.where(probabilities >= threshold)[0]
    if len(predicted_classes_indices) == 0:
        print("No classes predicted above the threshold.")
        return
    
    # Get similarity scores for each branch
    num_prototypes_1d = model_1d.num_prototypes
    num_prototypes_2d_partial = model_2d_partial.num_prototypes
    num_prototypes_2d_global = model_2d_global.num_prototypes
    
    sim1d = similarity_scores[:num_prototypes_1d]
    sim2d_partial = similarity_scores[num_prototypes_1d:num_prototypes_1d + num_prototypes_2d_partial]
    sim2d_global = similarity_scores[num_prototypes_1d + num_prototypes_2d_partial:]
    
    for class_idx in predicted_classes_indices:
        class_name = label_names[class_idx]
        print(f"\n--- Top Prototypes for Predicted Class: {class_name} (Prob: {probabilities[class_idx]:.4f}) ---")
        
        # Check each branch for relevant prototypes
        branch_contributions = []
        
        # 1D branch
        relevant_prototypes_1d = []
        for proto_id_str, meta in prototype_metadata1.items():
            if meta.get('prototype_class') == class_name:
                proto_idx = int(proto_id_str)
                if proto_idx < len(sim1d):
                    relevant_prototypes_1d.append({
                        'branch': '1D Rhythm',
                        'proto_id': proto_idx,
                        'score': sim1d[proto_idx],
                        'prototype_class': meta.get('prototype_class', 'N/A'),
                        'true_labels': meta.get('true_labels', [])
                    })
        
        # 2D Partial branch
        relevant_prototypes_2d_partial = []
        for proto_id_str, meta in prototype_metadata3.items():
            if meta.get('prototype_class') == class_name:
                proto_idx = int(proto_id_str)
                if proto_idx < len(sim2d_partial):
                    relevant_prototypes_2d_partial.append({
                        'branch': '2D Partial',
                        'proto_id': proto_idx,
                        'score': sim2d_partial[proto_idx],
                        'prototype_class': meta.get('prototype_class', 'N/A'),
                        'true_labels': meta.get('true_labels', [])
                    })
        
        # 2D Global branch
        relevant_prototypes_2d_global = []
        for proto_id_str, meta in prototype_metadata4.items():
            if meta.get('prototype_class') == class_name:
                proto_idx = int(proto_id_str)
                if proto_idx < len(sim2d_global):
                    relevant_prototypes_2d_global.append({
                        'branch': '2D Global',
                        'proto_id': proto_idx,
                        'score': sim2d_global[proto_idx],
                        'prototype_class': meta.get('prototype_class', 'N/A'),
                        'true_labels': meta.get('true_labels', [])
                    })
        
        # Combine and sort all prototypes
        all_prototypes = (relevant_prototypes_1d + relevant_prototypes_2d_partial + 
                         relevant_prototypes_2d_global)
        all_prototypes.sort(key=lambda x: x['score'], reverse=True)
        
        if not all_prototypes:
            print(f"  No prototypes found directly associated with '{class_name}'.")
            continue
        
        for i, p_info in enumerate(all_prototypes[:top_n]):
            true_label_names = [label_names[i] for i, v in enumerate(p_info['true_labels']) if v == 1.0]
            print(f"  {i+1}. {p_info['branch']} Prototype {p_info['proto_id']} "
                  f"(Prototype Class: {p_info['prototype_class']}, "
                  f"True Labels: {', '.join(true_label_names)}) – Score: {p_info['score']:.4f}")

def save_fusion_inference_results(ecg_id, probabilities, similarity_scores, label_names, 
                                 prototype_metadata1, prototype_metadata3, prototype_metadata4,
                                 model_1d, model_2d_partial, model_2d_global, 
                                 output_file="fusion_inference_results.json"):
    """
    Save fusion inference results to a JSON file.
    """
    results = {
        "ecg_id": int(ecg_id) if isinstance(ecg_id, (int, float)) else str(ecg_id),
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_config": {
            "model_type": "fusion",
            "fusion_weights": FUSION_WEIGHTS,
            "fusion_weights1": FUSION_WEIGHTS1,
            "fusion_weights3": FUSION_WEIGHTS3,
            "fusion_weights4": FUSION_WEIGHTS4
        },
        "predictions": {
            label_names[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        "top_predictions": [
            {"label": label_names[i], "probability": float(prob)} 
            for i, prob in sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)[:10]
        ],
        "branch_contributions": {
            "1d_rhythm": {
                "num_prototypes": model_1d.num_prototypes,
                "similarity_scores": similarity_scores[:model_1d.num_prototypes].tolist()
            },
            "2d_partial": {
                "num_prototypes": model_2d_partial.num_prototypes,
                "similarity_scores": similarity_scores[model_1d.num_prototypes:model_1d.num_prototypes + model_2d_partial.num_prototypes].tolist()
            },
            "2d_global": {
                "num_prototypes": model_2d_global.num_prototypes,
                "similarity_scores": similarity_scores[model_1d.num_prototypes + model_2d_partial.num_prototypes:].tolist()
            }
        }
    }
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fusion model inference and prototype analysis.")
    parser.add_argument('--show-prototypes', action='store_true', help='Show Top Prototypes and Similarity Scores for Each Class')
    parser.add_argument('--sanity-check', action='store_true', help='Run model performance sanity check on test set')
    parser.add_argument('--show-metadata', action='store_true', help='Print first 50 lines of metadata JSON')
    parser.add_argument('--check-labels', action='store_true', help='Check for label mismatch between CSV and inference label_names')
    parser.add_argument('--save-results', action='store_true', help='Save inference results to JSON file')
    parser.add_argument('--output-file', type=str, default='fusion_inference_results.json', help='Output file path for saved results')
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Fusion Label Mappings ---
    label_mappings = load_fusion_label_mappings()
    label_names = label_mappings["all"]
    num_classes = len(label_names)
    print(f"Loaded {num_classes} fusion labels")

    # --- Load Prototype Metadata for Each Branch ---
    with open(METADATA_JSON1, 'r') as f:
        prototype_metadata1 = json.load(f)
    with open(METADATA_JSON3, 'r') as f:
        prototype_metadata3 = json.load(f)
    with open(METADATA_JSON4, 'r') as f:
        prototype_metadata4 = json.load(f)

    # --- Load Fusion Model ---
    fusion_model, model_1d, model_2d_partial, model_2d_global = load_fusion_model()
    fusion_model = fusion_model.to(device)
    fusion_model.eval()

    # --- Get Test Data Loader ---
    # Create a simple args object for the dataloader
    class Args:
        def __init__(self):
            self.batch_size = 1
            self.sampling_rate = 100
            self.standardize = False
            self.remove_baseline = True
            self.num_workers = 0
    
    args_dataloader = Args()
    # Don't use return_sample_ids=True for fusion dataloader to avoid unpacking issues
    _, _, test_loader, _ = get_fusion_dataloaders(args_dataloader, return_sample_ids=False)

    if args.show_metadata:
        print(f"\n--- First 50 lines of {METADATA_JSON1} ---")
        with open(METADATA_JSON1, 'r') as f:
            for i, line in enumerate(f):
                print(line.rstrip())
                if i + 1 >= 50:
                    break

    # --- Select a Sample for Inference ---
    # For fusion, we'll use a simple approach to get samples
    print("\n--- Running Fusion Inference ---")
    
    # Get a few samples from the test loader
    test_iter = iter(test_loader)
    for sample_idx in range(5):  # Process 5 samples
        try:
            batch = next(test_iter)
            
            # Fusion dataloader returns (inputs, labels_dict)
            X_sample, labels_dict = batch
            X_sample = X_sample[0]  # Take first sample
            ecg_id = sample_idx
            
            print(f"\n--- Running Inference for Sample {ecg_id} ---")
            X_sample = X_sample.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = fusion_model(X_sample)
                probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
                
                # Get similarity scores from each branch
                x1d = X_sample.squeeze(1)
                _, _, sim1d = model_1d(x1d)
                _, _, sim2d_partial = model_2d_partial(X_sample)
                _, _, sim2d_global = model_2d_global(X_sample)
                
                # Concatenate similarity scores
                similarity_scores = torch.cat([sim1d, sim2d_partial, sim2d_global], dim=1).cpu().numpy().flatten()

            # --- Display Overall Predictions ---
            print("\n--- Fusion Model Overall Predictions (Top 10) ---")
            sorted_predictions = sorted(zip(label_names, probabilities), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_predictions[:10]:
                print(f"  {label}: {prob:.4f}")

            # --- Display True Labels (if available) ---
            if 'full' in labels_dict:
                y_sample = labels_dict['full'][0]  # Take first sample
                true_labels_indices = np.where(y_sample.cpu().numpy() == 1)[0]
                true_label_names = [label_names[i] for i in true_labels_indices]
                print(f"\n--- True Labels for Sample {ecg_id} ---")
                if true_label_names:
                    print(f"  {', '.join(true_label_names)}")
                else:
                    print("  No true labels (all zeros) for this sample.")

            # --- Display Fusion Prototypes for Predicted Classes ---
            display_fusion_prototypes(
                probabilities,
                similarity_scores,
                label_names,
                prototype_metadata1,
                prototype_metadata3,
                prototype_metadata4,
                model_1d,
                model_2d_partial,
                model_2d_global,
                threshold=0.5
            )

            # --- Save Inference Results ---
            if args.save_results:
                save_fusion_inference_results(
                    ecg_id, probabilities, similarity_scores, label_names,
                    prototype_metadata1, prototype_metadata3, prototype_metadata4,
                    model_1d, model_2d_partial, model_2d_global,
                    output_file=args.output_file
                )
                
        except StopIteration:
            break
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue

    print("\n--- Fusion Inference Complete ---") 
