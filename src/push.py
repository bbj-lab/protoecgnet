import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ecg_utils import load_label_mappings, plot_ecg

def log_ecg_to_tensorboard_1d(
    logger,
    ecg_signal,
    prototype_idx,
    class_1,
    class_2,
    similarity_score,
    save_dir,
    save_val=False,
    ecg_id=None,
    true_labels=None,
    label_mapping=None,
    extra_rhythm=None,
    highlight_rhythm=True,
):
    """
    Logs ECG waveform using printed 12-lead layout and saves it in a structured directory.
    """
    import matplotlib.pyplot as plt
    
    # Ensure correct shape 
    ecg_signal = np.squeeze(ecg_signal)
    if ecg_signal.shape == (12, 1000):
        ecg_signal = ecg_signal.T
    assert ecg_signal.shape == (1000, 12), f"Unexpected shape: {ecg_signal.shape}"


    display_id = ecg_id if ecg_id is not None else f"Prototype {prototype_idx}"

    if true_labels is not None and label_mapping is not None:
        true_label_names = [label_mapping[i] for i in np.where(true_labels == 1)[0]]
    else:
        true_label_names = None
    
    # Define how many rhythm strips (for highlighting part)
    num_rhythm_strips = 1

    # Plot ECG
    fig = plot_ecg(
        raw_ecg=ecg_signal,
        sampling_rate=100,
        ecg_id=display_id,
        true_labels=true_label_names,
        prototype_labels=[class_1, class_2] if class_2 else [class_1],
        rhythm_strip1='II', #r1,
        rhythm_strip2=None,
        rhythm_strip3=None,
        prototype_idx=prototype_idx, 
        similarity_score=similarity_score,
    )
    
    # Highlight bottom rhythm strip 
    if highlight_rhythm:
        ax = fig.axes[0]
        sampling_rate = 100
        highlight_start = 0
        highlight_end = ecg_signal.shape[0] / sampling_rate

        row_spacing_mm = 50
        stacked_leads = [['I', 'aVR', 'V1', 'V4'],
                        ['II', 'aVL', 'V2', 'V5'],
                        ['III', 'aVF', 'V3', 'V6']]
        row_baseline = len(stacked_leads + [None]) * row_spacing_mm - row_spacing_mm / 2

        highlight_y_center = row_baseline - len(stacked_leads) * row_spacing_mm
        highlight_y_top = highlight_y_center + 0.5 * row_spacing_mm
        highlight_y_bottom = highlight_y_center - 0.5 * row_spacing_mm

        ax.axhspan(
            highlight_y_bottom, highlight_y_top,
            xmin=highlight_start / 10,
            xmax=highlight_end / 10,
            color='blue',
            alpha=0.2,
            zorder=1
        )

    class_folder = f"{class_1}_{class_2}" if class_2 else class_1
    save_path = os.path.join(save_dir, f"Dual-Class/{class_folder}" if class_2 else class_folder,
                             f"Prototype_{prototype_idx}_ECG.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save locally
    if save_val:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # TensorBoard Logging 
    tag = f"Prototypes/Dual-Class/{class_folder}/Prototype_{prototype_idx}" if class_2 else \
          f"Prototypes/Class/{class_1}/Prototype_{prototype_idx}"
    logger.experiment.add_figure(tag, fig)

    plt.close(fig)
    
    if extra_rhythm is not None:
        import matplotlib.pyplot as plt

        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_map = {name: i for i, name in enumerate(lead_names)}
        if extra_rhythm not in lead_map:
            raise ValueError(f"Invalid rhythm lead '{extra_rhythm}'")

        lead_idx = lead_map[extra_rhythm]
        t = np.linspace(0, 10, ecg_signal.shape[0])
        signal = ecg_signal[:, lead_idx]

        fig_rhythm, ax = plt.subplots(figsize=(10, 2))

        # Draw grid: small pink lines every 0.04s, bold red every 0.2s
        for x in np.arange(0, 10.04, 0.04):
            ax.axvline(x, color='pink', linewidth=0.5, zorder=0)
        for x in np.arange(0, 10.2, 0.2):
            ax.axvline(x, color='red', linewidth=1.0, zorder=0)

        y_min, y_max = signal.min(), signal.max()
        step = 0.1  # ~1 mm assuming 10 mm/mV
        for y in np.arange(y_min - 0.2, y_max + 0.2, step):
            is_large = np.round(y * 10) % 5 == 0
            ax.axhline(y, color='pink' if not is_large else 'red',
                       linewidth=0.5 if not is_large else 1.0, zorder=0)

        # Plot signal
        ax.plot(t, signal, color='black', linewidth=1.0)
        ax.set_title(f"ECG {ecg_id} | Lead {extra_rhythm} Rhythm Strip", fontsize=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, 10)

        rhythm_filename = save_path.replace(".png", f"_rhythm_{extra_rhythm}.png")
        if save_val:
            fig_rhythm.savefig(rhythm_filename, bbox_inches='tight', dpi=300)
        plt.close(fig_rhythm)



def log_ecg_to_tensorboard_2d(
    logger,
    ecg_full,
    prototype_segment,
    prototype_idx,
    class_1,
    class_2,
    similarity_score,
    ecg_plot_dir,
    save_val=False,
    ecg_id=None, 
    true_labels=None,
    label_mapping=None,
    extra_segment=True,
):
    """
    Logs the full raw ECG using a printed-style layout and highlights the prototype match region.
    """
    import matplotlib.pyplot as plt
    # Reshape ecg_full to (1000, 12)
    ecg_full = np.squeeze(ecg_full)

    if ecg_full.shape == (12, 1000):
        ecg_full = ecg_full.T
    elif ecg_full.shape != (1000, 12):
        raise ValueError(f"[ERROR] Expected ECG shape (1000, 12) or (12, 1000), but got {ecg_full.shape}")

    # Estimate prototype match region 
    proto_len = prototype_segment.shape[1]

    if (proto_len == 1000):
        extra_segment=False

    match_scores = []
    for lead in range(ecg_full.shape[1]):
        sim = np.correlate(ecg_full[:, lead], prototype_segment[lead], mode="valid")
        match_scores.append(sim)
    match_scores = np.array(match_scores)
    best_idx = np.argmax(np.mean(match_scores, axis=0))

    if true_labels is not None and label_mapping is not None:
        true_label_names = [label_mapping[i] for i in np.where(true_labels == 1)[0]]
    else:
        true_label_names = None

    sampling_rate = 100
    highlight_start = best_idx / sampling_rate
    highlight_end = min(highlight_start + proto_len / sampling_rate, 10.0)

    fig = plot_ecg(
        raw_ecg=ecg_full,
        sampling_rate=sampling_rate,
        ecg_id=ecg_id, #if ecg_id is not None else f"Prototype {prototype_idx}",
        true_labels=true_label_names,
        prototype_labels=[class_1, class_2] if class_2 else [class_1],
        rhythm_strip1='II',
        rhythm_strip2=None, #'V1',
        rhythm_strip3=None, #'V6',
        prototype_idx=prototype_idx, 
        similarity_score=similarity_score,
    )

    # Highlight matched region 
    ax = fig.axes[0]
    if prototype_segment.shape[1] < 1000: #only highlight for partial prototypes
        ax.axvspan(highlight_start, highlight_end, color='blue', alpha=0.2, zorder=1)

    # Save locally
    if save_val:
        class_folder = f"{class_1}_{class_2}" if class_2 else class_1
        save_path = os.path.join(ecg_plot_dir, f"{class_folder}/Prototype_{prototype_idx}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    # TensorBoard Logging 
    tag = f"Prototypes/{class_1 if not class_2 else f'{class_1}_{class_2}'}/Prototype_{prototype_idx}"
    logger.experiment.add_figure(tag, fig)
    plt.close(fig)
    
    if extra_segment:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order = [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6']
        ]
        lead_name_to_idx = {name: i for i, name in enumerate(lead_names)}
        t_segment = np.linspace(highlight_start, highlight_end, prototype_segment.shape[1])

        fig_seg, axs = plt.subplots(3, 4, figsize=(12, 6))  # adjust size as needed
        axs = axs.flatten()

        for i, (row) in enumerate(lead_order):
            for j, lead in enumerate(row):
                idx = i * 4 + j
                lead_idx = lead_name_to_idx[lead]
                ax = axs[idx]
                ax.plot(t_segment, prototype_segment[lead_idx], color='black', linewidth=1.0)
                ax.set_ylabel(lead, rotation=0, labelpad=20, fontsize=10, fontweight='bold')
                ax.set_xlim(t_segment[0], t_segment[-1])
                ax.set_ylim(np.min(prototype_segment[lead_idx]) - 0.5, np.max(prototype_segment[lead_idx]) + 0.5)
                ax.tick_params(left=False, labelleft=False)

                # Draw ECG grid
                for x in np.arange(t_segment[0], t_segment[-1] + 0.04, 0.04):
                    ax.axvline(x, color='pink', linewidth=0.3, alpha=0.5)
                for x in np.arange(t_segment[0], t_segment[-1], 0.2):
                    ax.axvline(x, color='red', linewidth=0.5, alpha=0.7)

        fig_seg.suptitle(f"Prototype {prototype_idx} | Highlighted Segment", fontsize=12)
        fig_seg.tight_layout()

        seg_filename = os.path.join(
            ecg_plot_dir,
            f"{class_1}_{class_2}" if class_2 else class_1,
            f"Prototype_{prototype_idx}_extra_segment.png"
        )
        os.makedirs(os.path.dirname(seg_filename), exist_ok=True)
        if save_val:
            fig_seg.savefig(seg_filename, bbox_inches='tight', dpi=300)
        plt.close(fig_seg)


def label_matches(label_vector, class_indices):
    return any(label_vector[i] == 1 for i in class_indices)

def push_prototypes1d(model, dataloader, save_dir, label_set, job_name, logger=None, device="cuda", custom_groups=False):
    """
    Push each prototype to the closest training feature and log results.

    Args:
        model: The prototype-based model.
        dataloader: DataLoader containing training data.
        save_dir: Directory to save prototype information.
        label_set: The chosen label set (superdiagnostic, subdiagnostic, etc.).
        job_name: Name of the current job (for saving purposes).
        logger: Logger for TensorBoard.
        device: Device to use ("cuda" or "cpu").
    """

    model.to(device)
    model.eval()

    label_mappings = load_label_mappings(custom_groups=custom_groups,
                                prototype_category=None if not custom_groups else int(label_set))
    label_set_names = label_mappings["custom"] if custom_groups else label_mappings[label_set]

    label_mapping = {i: label_set_names[i] for i in range(len(label_set_names))}

    # Define logging directory 
    ecg_plot_dir = os.path.join(save_dir, f"prototype_ecgs_{job_name}")
    os.makedirs(ecg_plot_dir, exist_ok=True)

    # Store best matches for each prototype
    num_prototypes = model.prototype_vectors.shape[0]
    best_scores = torch.full((num_prototypes,), -np.inf, device=device)
    best_features = torch.zeros_like(model.prototype_vectors, device=device)
    prototype_metadata = {}
    class_distribution = {}
    final_prototype_assignments = {}

    print(f"Model latent space type: {model.latent_space_type}")

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch, sample_ids) in enumerate(tqdm(dataloader, desc="Pushing Prototypes")):
            X_batch = X_batch.to(device)

            # Extract feature representations & prototype similarity scores
            features, activations = model.push_forward(X_batch)

            for j in range(num_prototypes):
                class_vector = model.prototype_class_identity[j].cpu().numpy()
                class_indices = np.where(class_vector == 1)[0]
                class_names = [label_mapping[idx] for idx in class_indices]

                for i in range(activations.shape[0]):  # loop over batch samples
                    if label_matches(y_batch[i].cpu().numpy(), class_indices):
                        score = activations[i, j]
                        if score > best_scores[j]:
                            best_scores[j] = score
                            best_features[j] = features[i]
                            ecg_id = int(sample_ids[i].cpu().item())

                            if len(class_names) == 1:
                                prototype_class = class_names[0]
                                second_class = None
                            elif len(class_names) == 2:
                                prototype_class, second_class = class_names
                            else:
                                print(f"[WARNING] Prototype {j} has an unexpected number of classes: {class_names}")
                                prototype_class, second_class = class_names[0], None

                            prototype_metadata[j] = {
                                "ecg_id": ecg_id,
                                "prototype_class": prototype_class,
                                "true_labels": y_batch[i].cpu().numpy().tolist(),
                                "class_type": "dual" if second_class else "single",
                                "similarity_score": float(score.cpu().numpy())
                            }
                            final_prototype_assignments[j] = {
                                "ecg_id": ecg_id,
                                "ecg_data": X_batch[i].cpu().numpy(),
                                "prototype_class": prototype_class,
                                "true_labels": y_batch[i].cpu().numpy(),
                                "class_type": "dual" if second_class else "single",
                                "similarity_score": float(score.cpu().numpy())
                            }
                            if second_class:
                                prototype_metadata[j]["second_class"] = second_class
                                final_prototype_assignments[j]["second_class"] = second_class

                    # Debugging print
                    #print(f"[DEBUG] Prototype {j} assigned to ECG {ecg_id} with class labels {class_names}")

    if logger is not None: # Log ECG plots to Tensorboard
        save_val = True  # Whether to save the .png files to the checkpoint directory too
        for j, assignment in final_prototype_assignments.items():
            log_ecg_to_tensorboard_1d(
                logger, 
                assignment["ecg_data"], 
                j, 
                assignment["prototype_class"], 
                assignment.get("second_class", None), 
                assignment["similarity_score"], 
                ecg_plot_dir, 
                save_val=save_val,
                ecg_id=assignment["ecg_id"],
                true_labels=assignment["true_labels"],
                label_mapping=label_mapping,
                extra_rhythm="II",
            )

    # After all batches, count prototypes per class
    for j, assignment in final_prototype_assignments.items():
        prototype_class = assignment["prototype_class"]
        second_class = assignment.get("second_class", None)

        # Ensure each prototype is only counted once per class
        if prototype_class not in class_distribution:
            class_distribution[prototype_class] = 0
        class_distribution[prototype_class] += 1

        if second_class:
            if second_class not in class_distribution:
                class_distribution[second_class] = 0
            class_distribution[second_class] += 1

    # Save prototype vectors
    model.prototype_vectors.data.copy_(best_features)
    print("Prototype projection completed.")

    # Save prototype metadata (ECG IDs, class info, etc.)
    metadata_save_path = os.path.join(save_dir, f"{job_name}_prototype_metadata.json")
    with open(metadata_save_path, "w") as f:
        json.dump(prototype_metadata, f, indent=4)
    print(f"Saved prototype metadata to {metadata_save_path}")

    # TensorBoard Logging
    if logger is not None:
        # Log individual prototype similarity scores
        for j in range(model.prototype_vectors.shape[0]):
            logger.experiment.add_scalar(f"Similarity/Prototype_{j}", best_scores[j].item())

        # Log per-class prototype counts
        for class_name, count in class_distribution.items():
            logger.experiment.add_scalar(f"Count/{class_name}", count)

        # Log overall best similarity mean
        logger.experiment.add_scalar("Best_Similarity_Mean", best_scores.mean().item())

        # Log best-matching ECG IDs
        for j, metadata in prototype_metadata.items():
            logger.experiment.add_text(f"ECG_ID/Prototype_{j}", f"ECG {metadata['ecg_id']} ({metadata['prototype_class']})")

def push_prototypes2d(model, dataloader, save_dir, label_set, job_name, logger=None, device="cuda", custom_groups=False):
    """
    Push each prototype to the closest training feature and log results.

    Args:
        model: The prototype-based model.
        dataloader: DataLoader containing training data.
        save_dir: Directory to save prototype information.
        label_set: The chosen label set (superdiagnostic, subdiagnostic, etc.).
        job_name: Name of the current job (for saving purposes).
        logger: Logger for TensorBoard.
        device: Device to use ("cuda" or "cpu").
    """

    model.to(device)
    model.eval()

    label_mappings = load_label_mappings(custom_groups=custom_groups,
                                prototype_category=None if not custom_groups else int(label_set))
    label_set_names = label_mappings["custom"] if custom_groups else label_mappings[int(label_set)]

    label_mapping = {i: label_set_names[i] for i in range(len(label_set_names))}

    # Define logging directory 
    ecg_plot_dir = os.path.join(save_dir, f"prototype_ecgs_{job_name}")
    os.makedirs(ecg_plot_dir, exist_ok=True)

    num_prototypes = model.prototype_vectors.shape[0]
    best_scores = torch.full((num_prototypes,), -np.inf, device=device)
    best_features = torch.zeros_like(model.prototype_vectors, device=device)

    # Compute raw time length based on proto_time_len
    time_ratio = 1000 / 32  # ensure this stays updated (should not change, but had to explicitely initialize it this way)
    raw_time_len = int(model.proto_time_len * time_ratio)  # Dynamic mapping
    #print(f"[DEBUG] raw_time_len: {raw_time_len}")
    best_segments = torch.zeros((num_prototypes, 12, raw_time_len), device=device)

    prototype_metadata = {}
    class_distribution = {}
    final_prototype_assignments = {}

    print(f"Model latent space type: {model.latent_space_type}")

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch, sample_ids) in enumerate(tqdm(dataloader, desc="Pushing Prototypes")):
            X_batch = X_batch.to(device)  # Shape: (batch, 1, 12, 1000)

            #print(f"[DEBUG] X_batch shape: {X_batch.shape}")
            batch_size, _, num_channels, signal_length = X_batch.shape  
            
            # Extract feature representations & prototype similarity scores
            features, activations = model.push_forward(X_batch)  # activations: (batch, num_prototypes, time_windows)
            _, _, _, feature_time = features.shape  # Feature map shape (batch, 512, 1, 32)

            for j in range(num_prototypes):
                class_vector = model.prototype_class_identity[j].cpu().numpy()
                class_indices = np.where(class_vector == 1)[0]  # Indices of target classes
                class_names = [label_mapping[idx] for idx in class_indices]

                for b in range(batch_size):
                    if any(y_batch[b, class_idx] == 1 for class_idx in class_indices):
                        score = activations[b, j].max()
                        if score > best_scores[j]:
                            best_scores[j] = score
                            max_ecg_idx = b
                            max_window_idx = torch.argmax(activations[b, j])

                            if model.proto_time_len is None:
                                best_features[j] = features[max_ecg_idx]
                                best_segments[j] = X_batch[max_ecg_idx]
                            else:
                                max_window_end = min(max_window_idx + model.proto_time_len, feature_time)
                                best_features[j] = features[max_ecg_idx, :, :, max_window_idx:max_window_end]

                                time_scaling_factor = signal_length / feature_time
                                raw_max_time_idx = int(max_window_idx * time_scaling_factor)
                                raw_max_time_idx = max(0, min(raw_max_time_idx, signal_length - raw_time_len))
                                raw_time_end = min(raw_max_time_idx + raw_time_len, signal_length)

                                extracted_segment = X_batch[max_ecg_idx, :, :, raw_max_time_idx:raw_time_end]
                                best_segments[j] = extracted_segment.view(12, raw_time_len)

                            ecg_id = int(sample_ids[max_ecg_idx].cpu().item())

                            # Assign class metadata correctly
                            if len(class_names) == 1:
                                prototype_class = class_names[0]
                                second_class = None
                            elif len(class_names) == 2:
                                prototype_class, second_class = class_names
                            else:
                                print(f"[WARNING] Prototype {j} has unexpected classes: {class_names}")
                                prototype_class, second_class = class_names[0], None

                            prototype_metadata[j] = {
                                "ecg_id": ecg_id,
                                "prototype_class": prototype_class,
                                "true_labels": y_batch[max_ecg_idx].cpu().numpy().tolist(),
                                "class_type": "dual" if second_class else "single",
                                "similarity_score": float(best_scores[j].cpu().numpy())
                            }
                            final_prototype_assignments[j] = {
                                "ecg_id": ecg_id,
                                "ecg_data": X_batch[max_ecg_idx].cpu().numpy(),  # Full ECG
                                "prototype_segment": best_segments[j].cpu().numpy(),  # Extracted prototype segment
                                "true_labels": y_batch[max_ecg_idx].cpu().numpy(),
                                "prototype_class": prototype_class,
                                "class_type": "dual" if second_class else "single",
                                "similarity_score": float(best_scores[j].cpu().numpy())
                            }
                            if second_class:
                                prototype_metadata[j]["second_class"] = second_class
                                final_prototype_assignments[j]["second_class"] = second_class

    # Log ECG plots to TensorBoard
    if logger is not None:
        save_val = True # Whether to save the .png files
        for j, assignment in final_prototype_assignments.items():
            log_ecg_to_tensorboard_2d(
                logger, 
                assignment["ecg_data"],  # Full ECG
                assignment["prototype_segment"],  # Best-matching ECG segment
                j, 
                assignment["prototype_class"], 
                assignment.get("second_class", None), 
                assignment["similarity_score"], 
                ecg_plot_dir, 
                save_val=save_val, 
                true_labels=assignment["true_labels"],
                label_mapping=label_mapping,
                ecg_id=assignment["ecg_id"],
                extra_segment=True,
            )

    # Save prototype vectors
    model.prototype_vectors.data.copy_(best_features)
    print("Prototype projection completed.")

    # Save prototype metadata
    metadata_save_path = os.path.join(save_dir, f"{job_name}_prototype_metadata.json")
    with open(metadata_save_path, "w") as f:
        json.dump(prototype_metadata, f, indent=4)
    print(f"Saved prototype metadata to {metadata_save_path}")

    # TensorBoard Logging
    if logger is not None:
        for j in range(model.prototype_vectors.shape[0]):
            logger.experiment.add_scalar(f"Similarity/Prototype_{j}", best_scores[j].item())

        for class_name, count in class_distribution.items():
            logger.experiment.add_scalar(f"Count/{class_name}", count)

        logger.experiment.add_scalar("Best_Similarity_Mean", best_scores.mean().item())

        for j, metadata in prototype_metadata.items():
            logger.experiment.add_text(f"ECG_ID/Prototype_{j}", f"ECG {metadata['ecg_id']} ({metadata['prototype_class']})")

