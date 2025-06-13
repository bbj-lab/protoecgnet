import pandas as pd
import numpy as np
import wfdb
import ast
import os
import torch
from sklearn.preprocessing import StandardScaler
import pickle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# FILE PATHS TO CHANGE
# HPC
DATASET_PATH = "/gpfs/data/bbj-lab/users/sethis/physionet.org/files/ptb-xl/1.0.3"
STANDARDIZATION_PATH = '/gpfs/data/bbj-lab/users/sethis/experiments/preprocessing'
SCP_GROUP_PATH = "scp_statementsRegrouped2.csv"
# SCP_GROUP_PATH can only be used when scp_statementsRegrouped2.csv is in the same directory
# as DATASET_PATH. SCP_GROUP_PATH_ABS allows us to provide a separate path for scp_statementsRegrouped2.csv
SCP_GROUP_PATH_ABS = "/home/chend5/protoecgnet/scp_statementsRegrouped2.csv"

def remove_baseline_wander(X, sampling_rate=100, cutoff=0.5, order=1):
    """
    Applies a high-pass Butterworth filter to remove baseline wander.
    Operates on numpy arrays shaped (N, 1000, 12) or (N, 12, 1000).
    No need to apply a low-pass filter if using 100 Hz data
    """
    b, a = butter(order, cutoff / (sampling_rate / 2), btype='high', analog=False)
    
    if X.ndim == 3 and X.shape[1] == 1000 and X.shape[2] == 12:
        # Shape: (N, 1000, 12)
        for i in range(X.shape[0]):
            for j in range(12):
                X[i, :, j] = filtfilt(b, a, X[i, :, j])
    elif X.ndim == 3 and X.shape[1] == 12 and X.shape[2] == 1000:
        # Shape: (N, 12, 1000)
        for i in range(X.shape[0]):
            for j in range(12):
                X[i, j, :] = filtfilt(b, a, X[i, j, :])
    else:
        raise ValueError(f"Unexpected shape for baseline wander correction: {X.shape}")
    
    return X

def plot_ecg(
    raw_ecg,
    sampling_rate=100,
    ecg_id=None,
    true_labels=None,
    prototype_labels=None,
    rhythm_strip1='II',
    rhythm_strip2=None,
    rhythm_strip3=None,
    prototype_idx=None,
    similarity_score=None
):
    import matplotlib.pyplot as plt
    import numpy as np

    if raw_ecg.shape == (12, 1000):
        raw_ecg = raw_ecg.T
    assert raw_ecg.shape == (1000, 12)

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_name_to_idx = {name: i for i, name in enumerate(lead_names)}

    rhythm_leads = []
    for lead in [rhythm_strip1, rhythm_strip2, rhythm_strip3]:
        if lead is not None:
            lead_str = lead if isinstance(lead, str) else lead[0]  
            rhythm_leads.append(lead_name_to_idx[lead_str])

    mm_per_mv = 10
    mm_per_sec = 25
    paper_speed = mm_per_sec
    gain = mm_per_mv

    duration = raw_ecg.shape[0] / sampling_rate
    stacked_leads = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6']
    ]

    num_rows = len(stacked_leads) + len(rhythm_leads)
    row_spacing_mm = 50

    fig_width_in = (duration * paper_speed) / 25.4
    fig_height_in = (num_rows * row_spacing_mm) / 25.4

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    ax.set_xlim(0, duration)
    ax.set_ylim(0, num_rows * row_spacing_mm)
    ax.axis('off')

    def draw_paper_grid():
        for x in np.arange(0, duration + 0.04, 0.04):
            ax.axvline(x, color='pink', linewidth=0.5, zorder=0)
        for y in np.arange(0, num_rows * row_spacing_mm + 0.1, 1):
            is_large = (y * 0.1) % 0.5 == 0
            ax.axhline(y, color='pink' if not is_large else 'red', linewidth=0.5 if not is_large else 1.0, zorder=0)
        for x in np.arange(0, duration + 0.2, 0.2):
            ax.axvline(x, color='red', linewidth=1.0, zorder=0)

    draw_paper_grid()

    label_fontsize = 12
    row_baseline = num_rows * row_spacing_mm - row_spacing_mm / 2

    for row_idx, lead_row in enumerate(stacked_leads):
        for col_idx, lead in enumerate(lead_row):
            lead_idx = lead_name_to_idx[lead]
            start = col_idx * 250
            end = (col_idx + 1) * 250
            signal = raw_ecg[start:end, lead_idx] * gain
            t = np.linspace(col_idx * 2.5, (col_idx + 1) * 2.5, 250)
            v_offset = row_baseline - row_idx * row_spacing_mm
            ax.plot(t, signal + v_offset, color='black', linewidth=1.0)
            ax.text(t[0] + 0.1, v_offset + 16, lead, fontsize=label_fontsize, fontweight='bold')

    # Rhythm leads: full 10 seconds
    t = np.linspace(0, 10, raw_ecg.shape[0])
    for i, lead_idx in enumerate(rhythm_leads):
        v_offset = row_baseline - (len(stacked_leads) + i) * row_spacing_mm
        signal = raw_ecg[:, lead_idx] * gain
        ax.plot(t, signal + v_offset, color='black', linewidth=1.0)
        ax.text(0.1, v_offset + 16, f"{lead_names[lead_idx]}", fontsize=label_fontsize, fontweight='bold')

    # Title
    title = f"ECG {ecg_id}"
    if true_labels:
        title += f" | True: {true_labels}"
    if prototype_idx is not None: 
        title += f" | Prototype Labels: {prototype_labels}"
        similarity_score = round(similarity_score, 3)
        title += f" | Similarity Score: {similarity_score}"
    fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()
    return fig

def standardize_signals(X_train, X_val, X_test, output_folder, mode):
    scaler_path = os.path.join(output_folder, f"standard_scaler_{mode}.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            ss = pickle.load(f)
    else:
        ss = StandardScaler()

        if mode == '1D':
            # Reshape to (num_samples * timepoints, leads) 
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])  # Shape (N * 1000, 12)
        elif mode == '2D':
            # Reshape to (num_samples * height * width, leads) 
            X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)  # (N, 12, H*W)
            X_train_flat = X_train_flat.transpose(1, 0, 2).reshape(X_train.shape[1], -1).T  # (12, N*H*W)

        ss.fit(X_train_flat)

        with open(scaler_path, "wb") as f:
            pickle.dump(ss, f)

    return apply_standardizer(X_train, ss, mode), apply_standardizer(X_val, ss, mode), apply_standardizer(X_test, ss, mode)

def apply_standardizer(X, ss, mode):
    if mode == '1D':
        X_shape = X.shape  # (N, 1000, 12)
        X_flat = X.reshape(-1, X.shape[-1])  # (N * 1000, 12) 
        X_std = ss.transform(X_flat)  
        return X_std.reshape(X_shape)  # Reshape back to (N, 1000, 12)

    elif mode == '2D':
        X_shape = X.shape  # (N, 12, H, W)
        X_flat = X.reshape(X.shape[0], X.shape[1], -1)  # (N, 12, H*W)
        X_flat = X_flat.transpose(1, 0, 2).reshape(X.shape[1], -1).T  # (12, N*H*W)
        X_std = ss.transform(X_flat)  
        X_std = X_std.T.reshape(X.shape[1], X.shape[0], -1).transpose(1, 0, 2)  
        return X_std.reshape(X_shape)  # Restore (N, 12, H, W)

def load_label_mappings(custom_groups=False, prototype_category=None):
    if custom_groups:
        if SCP_GROUP_PATH_ABS is not None:
            label_df = pd.read_csv(SCP_GROUP_PATH_ABS, index_col=0)
        else:
            label_df = pd.read_csv(os.path.join(DATASET_PATH, SCP_GROUP_PATH), index_col=0)
        
        # Ensure the new column exists
        assert "prototype_category" in label_df.columns, "Missing 'prototype_category' column in regrouped SCP file."

        # Filter by category if specified
        if prototype_category in [1, 2, 3, 4]:
            custom_labels = label_df[label_df["prototype_category"] == prototype_category].index.tolist()
        else:
            custom_labels = label_df.index.tolist()

        print(f"Loaded {len(custom_labels)} custom group labels (Category: {prototype_category})")

        return {
            "custom": custom_labels,
        }
    else: 
        label_df = pd.read_csv(os.path.join(DATASET_PATH, "scp_statements.csv"), index_col=0)

        # Extract mappings from SCP codes
        scp_to_superclass = label_df["diagnostic_class"].dropna().to_dict()  # SCP → Superclass
        scp_to_subclass = label_df["diagnostic_subclass"].dropna().to_dict()  # SCP → Subclass

        superdiagnostic_labels = label_df[label_df["diagnostic_class"].notna()]["diagnostic_class"].unique().tolist()
        subdiagnostic_labels = label_df[label_df["diagnostic_subclass"].notna()]["diagnostic_subclass"].unique().tolist()
        all_labels = label_df.index.tolist() 
        
        diagnostic_labels = label_df[label_df["diagnostic"] == 1.0].index.tolist()
        form_labels = label_df[label_df["form"] == 1.0].index.tolist()
        rhythm_labels = label_df[label_df["rhythm"] == 1.0].index.tolist()

        print(f"Loaded Label Mappings - Super: {len(superdiagnostic_labels)}, Sub: {len(subdiagnostic_labels)}, "
            f"All: {len(all_labels)}, Diagnostic: {len(diagnostic_labels)}, Form: {len(form_labels)}, "
            f"Rhythm: {len(rhythm_labels)}")

        return {
            "superdiagnostic": superdiagnostic_labels,
            "subdiagnostic": subdiagnostic_labels,
            "all": all_labels,
            "diagnostic": diagnostic_labels,
            "form": form_labels,
            "rhythm": rhythm_labels,
            "scp_to_superclass": scp_to_superclass,
            "scp_to_subclass": scp_to_subclass
        }


def load_raw_data(sampling_rate, label_type, df, custom_groups=False):
    label_mappings = load_label_mappings(custom_groups=custom_groups,
                                prototype_category=None if not custom_groups else int(label_type))
    selected_labels = label_mappings["custom"] if custom_groups else label_mappings[label_type]

    if not custom_groups: 
        scp_to_superclass = label_mappings["scp_to_superclass"]
        scp_to_subclass = label_mappings["scp_to_subclass"]

    # Define which column to use for label mapping
    if label_type == "superdiagnostic":
        label_map = scp_to_superclass
    elif label_type == "subdiagnostic":
        label_map = scp_to_subclass
    else:
        label_map = None  # "all", "diagnostic", "form", and "rhythm" directly use SCP codes

    # Containers
    signals = []
    valid_indices = []
    labels = []
    filtered_out = 0  # Track filtered samples

    for idx, filename in enumerate(df.filename_lr if sampling_rate == 100 else df.filename_hr):
        full_path = os.path.join(DATASET_PATH, filename)

        if not os.path.exists(full_path + ".dat"):
            print(f"Missing file: {full_path}. Skipping...")
            continue

        try:
            # Load ECG signal
            signal, _ = wfdb.rdsamp(full_path)

            # Extract SCP codes
            label_dict = df.iloc[idx].scp_codes  
            label_vector = np.zeros(len(selected_labels))
            has_valid_label = False

            contains_norm = False 
            for scp_code in label_dict.keys():
                if custom_groups:
                    # Only use SCP codes directly if they are in the selected custom group
                    if scp_code in selected_labels:
                        label_vector[selected_labels.index(scp_code)] = 1
                        has_valid_label = True
                elif label_type in ["all", "diagnostic", "form", "rhythm"]:
                    if scp_code in selected_labels:
                        label_vector[selected_labels.index(scp_code)] = 1
                        has_valid_label = True
                else:
                    # Only if label_map is not None
                    mapped_label = label_map.get(scp_code, None)
                    if mapped_label in selected_labels:
                        label_vector[selected_labels.index(mapped_label)] = 1
                        has_valid_label = True
                        if mapped_label == "NORM":
                            contains_norm = True
            if contains_norm and "NORM" in selected_labels:
                label_vector[selected_labels.index("NORM")] = 1
                has_valid_label = True


            # If at least one valid label is found, add to the dataset (only for superdiagnostic/subdiagnostic groupings)
            if label_type in ["superdiagnostic", "subdiagnostic"]:
                if has_valid_label:
                    signals.append(signal)
                    valid_indices.append(df.iloc[idx].name)
                    labels.append(label_vector)
                else:
                    filtered_out += 1
            else: 
                signals.append(signal)
                valid_indices.append(df.iloc[idx].name)
                labels.append(label_vector)

        except Exception as e:
            print(f"Error loading {full_path}: {e}")

    X = np.array(signals)
    y = np.array(labels)

    print(f"Final retained samples: {X.shape[0]}/{len(df)}")

    return X, y, valid_indices

 #Dataset Classes 
class PTBXL_Dataset_1D(Dataset):
    def __init__(self, X, y, sample_ids=None, return_sample_ids=False):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, 12, 1000)
        self.y = torch.tensor(y, dtype=torch.float32) 
        self.sample_ids = sample_ids if sample_ids is not None else np.arange(len(y)) 
        self.return_sample_ids = return_sample_ids 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.return_sample_ids: 
            return self.X[idx], self.y[idx], self.sample_ids[idx]
        else: 
            return self.X[idx], self.y[idx]

class PTBXL_Dataset_2D(Dataset):
    def __init__(self, X, y, sample_ids=None, return_sample_ids=False):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2)  # (N, 1, 12, 1000)
        self.y = torch.tensor(y, dtype=torch.float32)  
        self.sample_ids = sample_ids if sample_ids is not None else np.arange(len(y)) 
        self.return_sample_ids = return_sample_ids 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.return_sample_ids: 
            return self.X[idx], self.y[idx], self.sample_ids[idx]
        else: 
            return self.X[idx], self.y[idx]

# DataLoader Function
def get_dataloaders(batch_size=32, mode="2D", sampling_rate=100, label_set="superdiagnostic", work_num=4, return_sample_ids=False, custom_groups=False, standardize=False, remove_baseline=False):
    df = pd.read_csv(os.path.join(DATASET_PATH, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x)) 
    
    # Split data using PTB-XL folds (Train: folds 1-8, Val: fold 9, Test: fold 10)
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    test_df = df[df.strat_fold == 10]
    
    X_train, y_train, train_sample_ids = load_raw_data(sampling_rate, label_set, train_df, custom_groups=custom_groups)
    X_val, y_val, val_sample_ids = load_raw_data(sampling_rate, label_set, val_df, custom_groups=custom_groups)
    X_test, y_test, test_sample_ids = load_raw_data(sampling_rate, label_set, test_df, custom_groups=custom_groups)

    # Compute class weights
    def compute_class_weights(y):
        class_counts = np.sum(y, axis=0)  
        total_samples = sum(class_counts)
        class_weights = torch.tensor([(total_samples - c) / c for c in class_counts], dtype=torch.float32)
        return torch.tensor(class_weights, dtype=torch.float32)

    class_weights = compute_class_weights(y_train)

    # Apply preprocessing (optional)
    output_folder = STANDARDIZATION_PATH
    os.makedirs(output_folder, exist_ok=True)

    if remove_baseline:
        X_train = remove_baseline_wander(X_train, sampling_rate=sampling_rate)
        X_val = remove_baseline_wander(X_val, sampling_rate=sampling_rate)
        X_test = remove_baseline_wander(X_test, sampling_rate=sampling_rate)

    if standardize: 
        X_train, X_val, X_test = standardize_signals(X_train, X_val, X_test, output_folder, mode)

    print("\n--- Data Summary ---")
    print(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples, Test set: {len(test_df)} samples")
    print(f"Loaded training data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Loaded validation data: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"Loaded test data: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Label distribution (Train): {np.sum(y_train, axis=0)}")
    print(f"Label distribution (Validation): {np.sum(y_val, axis=0)}")
    print(f"Label distribution (Test): {np.sum(y_test, axis=0)}")
    print(f"Training set class weights: {class_weights}")
    
    train_dataset = PTBXL_Dataset_1D(X_train, y_train, train_sample_ids, return_sample_ids) if mode == '1D' else \
                    PTBXL_Dataset_2D(X_train, y_train, train_sample_ids, return_sample_ids)
    val_dataset = PTBXL_Dataset_1D(X_val, y_val, val_sample_ids, return_sample_ids) if mode == '1D' else \
                  PTBXL_Dataset_2D(X_val, y_val, val_sample_ids, return_sample_ids)
    test_dataset = PTBXL_Dataset_1D(X_test, y_test, test_sample_ids, return_sample_ids) if mode == '1D' else \
                   PTBXL_Dataset_2D(X_test, y_test, test_sample_ids, return_sample_ids)

    #Get validation set class weights for weighted random sampler
    val_class_weights = compute_class_weights(y_val) 
    sample_weights = np.dot(y_val, val_class_weights.numpy())  # Assign sample weights based on label presence
    val_sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_val), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=work_num)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=work_num)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=work_num)
    
    return train_loader, val_loader, test_loader, class_weights
