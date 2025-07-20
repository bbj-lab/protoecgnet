import os
import optuna
import torch
import torch.distributed as dist
import argparse
import pytorch_lightning as pl
from training_functions import train_model, test_model, seed_everything, save_study, load_study
from ecg_utils import get_dataloaders, load_label_mappings
from backbones import (
        resnet1d18, resnet1d34, resnet1d50, resnet1d101, resnet1d152, 
        resnet18, resnet34, resnet50, resnet101, resnet152
)
from proto_models1D import ProtoECGNet1D
from proto_models2D import ProtoECGNet2D
from fusion import FusionProtoClassifier
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from fusion import FusionProtoClassifier, load_fusion_label_mappings, get_fusion_dataloaders

seed_everything(42)
torch.set_float32_matmul_precision("high")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--job_name', type=str, required=True, help="Job name for tuning experiments")
parser.add_argument('--epochs', type=int, default=20, help="Number of epochs per trial")
parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size (can be tuned)")
parser.add_argument('--use_class_weights', type=str2bool, default=True)
parser.add_argument('--checkpoint_dir', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints', help="Directory for saving model checkpoints")
parser.add_argument('--log_dir', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/logs', help="Directory for TensorBoard logs")
parser.add_argument('--test_dir', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/test_results', help="Directory for test results")
parser.add_argument('--study_dir', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/optuna_studies', help="Directory for Optuna study")
parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained model weights')
parser.add_argument('--training_stage', type=str, choices=['feature_extractor', 'prototypes', 'joint', 'projection', 'classifier', 'fusion'], required=True)
parser.add_argument('--l1', type=float, default=1e-4)
parser.add_argument('--lam_clst', type=float, default=0.8)
parser.add_argument('--lam_sep', type=float, default=0.08)
parser.add_argument('--lam_spars', type=float, default=0.0001)
parser.add_argument('--lam_div', type=float, default=100)
parser.add_argument('--lam_cnrst', type=float, default=0.4)
parser.add_argument('--proto_dim', type=int, default=512, help='Dimension of prototype vectors')
parser.add_argument('--proto_time_len', type=int, default=8, help='Time dimension of prototype vectors')
parser.add_argument('--prototype_activation_function', type=str, choices=['log', 'linear'], default='log')
parser.add_argument('--latent_space_type', type=str, choices=['l2', 'arc'], default='arc')
parser.add_argument('--add_on_layers_type', type=str, choices=['linear', 'other', 'identity'], default='linear')
parser.add_argument('--class_specific', type=str2bool, default=True, help='Whether to use class-specific prototypes')
parser.add_argument('--last_layer_connection_weight', type=float, default=1.0, help='Weight of last layer connection in the model')
parser.add_argument('--m', type=float, default=0.05, help='Margin for prototype learning')
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to use for projection (cuda or cpu)")
parser.add_argument('--sampling_rate', type=int, choices=[100, 500], default=100, help="ECG sampling rate")
parser.add_argument('--label_set', type=str, choices=['superdiagnostic', 'subdiagnostic', 'all', 'diagnostic', 'form', 'rhythm', '1', '2', '3', '4'], default='superdiagnostic')
parser.add_argument('--num_workers', type=int, default=4, help="Number of data loading workers")
parser.add_argument('--dimension', type=str, choices=['1D', '2D'], required=True, help='Specify whether the model is 1D or 2D')
parser.add_argument('--backbone', type=str, choices=[
    'resnet1d18', 'resnet1d34', 'resnet1d50', 'resnet1d101', 'resnet1d152',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
], required=True, help='Specify the backbone architecture')
parser.add_argument('--custom_groups', type=str2bool, default=False, help='Flag to use custom label groupings')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--standardize', type=str2bool, default=False, help='Whether to standardize input ECG signals')
parser.add_argument('--remove_baseline', type=str2bool, default=True, help='Whether to remove baseline wander from input ECG signals (high-pass filter)')

parser.add_argument('--fusion_weights1', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat1_new_redo1_proj1/cat1_new_redo1_proj1_projection.pth', help='Path to pretrained model weights for category 1')
parser.add_argument('--fusion_weights3', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat3_2D_redo1_tunejoint1_trial96_proj/cat3_2D_redo1_tunejoint1_trial96_proj_projection.pth', help='Path to pretrained model weights for category 3')
parser.add_argument('--fusion_weights4', type=str, default='/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints/cat4_2D_redo1_tunejoint1_trial5_proj/cat4_2D_redo1_tunejoint1_trial5_proj_projection.pth', help='Path to pretrained model weights for category 4')
    
parser.add_argument('--fusion_backbone1', type=str, default='resnet1d18', help='Backbone for 1D rhythm model (category 1)')
parser.add_argument('--fusion_backbone3', type=str, default='resnet18', help='Backbone for 2D partial morphology model (category 3)')
parser.add_argument('--fusion_backbone4', type=str, default='resnet18', help='Backbone for 2D global model (category 4)')

parser.add_argument('--fusion_proto_dim1', type=int, default=512, help='Prototype dimension for 1D rhythm model (category 1)')
parser.add_argument('--fusion_proto_dim3', type=int, default=512, help='Prototype dimension for 2D partial morphology model (category 3)')
parser.add_argument('--fusion_proto_dim4', type=int, default=512, help='Prototype dimension for 2D global model (category 4)')

parser.add_argument('--fusion_single_ppc1', type=int, default=5, help='Single-class prototypes per class for category 1') 
parser.add_argument('--fusion_single_ppc3', type=int, default=18, help='Single-class prototypes per class for category 3')
parser.add_argument('--fusion_single_ppc4', type=int, default=7, help='Single-class prototypes per class for category 4')

parser.add_argument('--fusion_joint_ppb1', type=int, default=0, help='Joint prototypes per border for category 1')
parser.add_argument('--fusion_joint_ppb3', type=int, default=0, help='Joint prototypes per border for category 3')
parser.add_argument('--fusion_joint_ppb4', type=int, default=0, help='Joint prototypes per border for category 4')
args = parser.parse_args()

# Define directories for this job's trials
job_checkpoint_dir = os.path.join(args.checkpoint_dir, args.job_name)
job_log_dir = os.path.join(args.log_dir, args.job_name)
job_test_dir = os.path.join(args.test_dir, args.job_name)
os.makedirs(job_checkpoint_dir, exist_ok=True)
os.makedirs(job_log_dir, exist_ok=True)
os.makedirs(job_test_dir, exist_ok=True)

study_save_path = os.path.join(args.study_dir, f"{args.job_name}_optuna_study.pkl")

# TensorBoard Logger Setup
def create_tensorboard_logger(trial_number):
    log_dir = os.path.join(job_log_dir, f"trial_{trial_number}")
    return TensorBoardLogger(save_dir=log_dir, name=f"trial_{trial_number}")

def find_unfinished_trials(study):
    """Find trials that were interrupted (neither completed nor pruned) and should be restarted."""
    unfinished_trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:  
            unfinished_trials.append(trial.number)
    return unfinished_trials

class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, trial, monitor="val_auc"):
        super().__init__(trial, monitor)

# Optuna Objective Function
def objective(trial):
    """Optuna objective function for hyperparameter tuning."""

    # Define hyperparameter search space
    trial_params = {
        "backbone": args.backbone, 
        "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),  
        "scheduler_type": trial.suggest_categorical("scheduler", [
            "ReduceLROnPlateau", "CosineAnnealingLR", "CyclicLR"
        ]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "l2": trial.suggest_float("l2", 1e-6, 1e-2, log=True), 
        "l1": trial.suggest_float("l1", 1e-6, 1e-2, log=True),  #args.l1
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),  
        "single_class_prototype_per_class": trial.suggest_int("single_class_prototype_per_class", 1, 20),
        "joint_prototypes_per_border": 0, 
        "proto_time_len": args.proto_time_len,
        "lam_clst": 0.004, 
        "lam_sep": 0.0004, 
        "lam_spars": 0,
        "lam_div": 250, 
        "lam_cnrst": 300,
    }

    # Print trial details
    print(f"Trial {trial.number} - Single Process using: {trial_params}")

    # Set unique trial name
    trial_name = f"trial_{trial.number}"
    trial_checkpoint_dir = os.path.join(job_checkpoint_dir, trial_name)
    os.makedirs(trial_checkpoint_dir, exist_ok=True)

    print("Loading label mappings...")
    label_mappings = load_label_mappings(
        custom_groups=args.custom_groups,
        prototype_category=int(args.label_set) if args.custom_groups else None
    )

    if args.custom_groups:
        num_classes = len(label_mappings["custom"])
    else:
        num_classes = len(label_mappings[args.label_set])

    if args.training_stage != 'fusion':
        train_loader, val_loader, test_loader, class_weights = get_dataloaders(
            batch_size=trial_params["batch_size"], mode=args.dimension, sampling_rate=args.sampling_rate, 
                label_set=args.label_set, work_num=args.num_workers, custom_groups=args.custom_groups,
                standardize=args.standardize, remove_baseline=args.remove_baseline,
        )

        if args.use_class_weights: 
            class_wts = class_weights
        else: 
            class_wts=None

    # Initialize model
    model = eval(trial_params["backbone"])(num_classes=num_classes, dropout=trial_params["dropout"])
    # Model selection
    print(f"Selecting model: {trial_params['backbone']} with dimension {args.dimension} for stage {args.training_stage}...")
    if args.training_stage == "feature_extractor":
        model = eval(trial_params["backbone"])(num_classes=num_classes, dropout=trial_params["dropout"])  # Load backbone directly
        if args.pretrained_weights: # Load pretrained weights if specified
            state_dict = torch.load(args.pretrained_weights, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {args.pretrained_weights}")
    elif args.training_stage != 'fusion':
        if args.dimension == '1D':
            model = ProtoECGNet1D(num_classes=num_classes, single_class_prototype_per_class=trial_params["single_class_prototype_per_class"], 
                                  joint_prototypes_per_border=trial_params["joint_prototypes_per_border"], proto_dim=args.proto_dim, 
                                  backbone=trial_params["backbone"], prototype_activation_function=args.prototype_activation_function, 
                                  latent_space_type=args.latent_space_type, add_on_layers_type=args.add_on_layers_type, 
                                  class_specific=args.class_specific, last_layer_connection_weight=args.last_layer_connection_weight, 
                                  m=args.m, custom_groups=args.custom_groups, label_set = args.label_set, dropout=trial_params["dropout"], pretrained_weights=args.pretrained_weights)
        elif args.dimension == '2D':
            model = ProtoECGNet2D(num_classes=num_classes, single_class_prototype_per_class=trial_params["single_class_prototype_per_class"], 
                                  joint_prototypes_per_border=trial_params["joint_prototypes_per_border"], proto_dim=args.proto_dim, 
                                  backbone=trial_params["backbone"], prototype_activation_function=args.prototype_activation_function, 
                                  proto_time_len=trial_params["proto_time_len"], latent_space_type=args.latent_space_type, add_on_layers_type=args.add_on_layers_type, 
                                  class_specific=args.class_specific, last_layer_connection_weight=args.last_layer_connection_weight, 
                                  m=args.m, custom_groups=args.custom_groups, label_set = args.label_set, dropout=trial_params["dropout"], pretrained_weights=args.pretrained_weights)
        else:
            raise ValueError(f"Unsupported model dimension: {args.dimension}")


    # Set up PyTorch Lightning Trainer
    logger = create_tensorboard_logger(trial.number)
    checkpoint_callback = ModelCheckpoint(
        dirpath=trial_checkpoint_dir,  
        filename='{epoch}-{val_auc:.4f}',
        monitor='val_auc',
        mode='max',
        save_top_k=1, #adjust if you don't want to save models during tuning
        save_last=False,
    )
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=5, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    pruning_callback = OptunaPruning(trial, monitor="val_auc")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, pruning_callback],
    )

    # Train Model 
    train_args = argparse.Namespace(
        job_name=trial_name, epochs=args.epochs, batch_size=trial_params["batch_size"], lr=trial_params["lr"], l2=trial_params["l2"], dropout=trial_params["dropout"],
        checkpoint_dir=job_checkpoint_dir, log_dir=job_log_dir, test_dir=job_test_dir, 
        save_top_k=1, patience=10, resume_checkpoint=False, pretrained_weights=args.pretrained_weights, training_stage=args.training_stage,
        l1=trial_params["l1"], lam_clst=trial_params["lam_clst"], lam_sep=trial_params["lam_sep"], lam_spars=trial_params["lam_spars"], lam_div=trial_params["lam_div"], lam_cnrst=trial_params["lam_cnrst"], proto_dim=args.proto_dim, prototype_activation_function=args.prototype_activation_function, 
        latent_space_type=args.latent_space_type, add_on_layers_type=args.add_on_layers_type, class_specific=args.class_specific, 
        last_layer_connection_weight=args.last_layer_connection_weight, m=args.m, device=args.device, 
        dimension=args.dimension, backbone=trial_params["backbone"], single_class_prototype_per_class=trial_params["single_class_prototype_per_class"], joint_prototypes_per_border=trial_params["joint_prototypes_per_border"],
        sampling_rate=args.sampling_rate, label_set=args.label_set, test_model=False, save_weights=False, 
        seed=args.seed, num_workers=args.num_workers, scheduler_type=trial_params["scheduler_type"], custom_groups=args.custom_groups,
    )

    if args.training_stage == "fusion":
        label_map = load_fusion_label_mappings()

        num_classes1 = len(label_map["1"])
        num_classes3 = len(label_map["3"])
        num_classes4 = len(label_map["4"])


        print("Loading pretrained 1D and 2D models for fusion...")

        print(f"Creating category 1 model with backbone {args.fusion_backbone1}...")
        model_1d = ProtoECGNet1D(
            num_classes=num_classes1,
            single_class_prototype_per_class=args.fusion_single_ppc1,
            joint_prototypes_per_border=args.fusion_joint_ppb1,
            proto_dim=args.fusion_proto_dim1,
            backbone=args.fusion_backbone1,
            prototype_activation_function=args.prototype_activation_function,
            latent_space_type=args.latent_space_type,
            add_on_layers_type=args.add_on_layers_type,
            class_specific=args.class_specific,
            last_layer_connection_weight=args.last_layer_connection_weight,
            m=args.m,
            dropout=trial_params["dropout"],
            custom_groups=True,
            label_set="1",
            pretrained_weights=args.fusion_weights1
        )

        print(f"1D model loaded. Creating category 3 model with backbone {args.fusion_backbone3}...")

        model_2d_partial = ProtoECGNet2D(
            num_classes=num_classes3,
            single_class_prototype_per_class=args.fusion_single_ppc3,
            joint_prototypes_per_border=args.fusion_joint_ppb3,
            proto_dim=args.fusion_proto_dim3,
            proto_time_len=args.proto_time_len,
            backbone=args.fusion_backbone3,
            prototype_activation_function=args.prototype_activation_function,
            latent_space_type=args.latent_space_type,
            add_on_layers_type=args.add_on_layers_type,
            class_specific=args.class_specific,
            last_layer_connection_weight=args.last_layer_connection_weight,
            m=args.m,
            dropout=trial_params["dropout"],
            custom_groups=True,
            label_set="3",
            pretrained_weights=args.fusion_weights3
        )

        print(f"2D partial model loaded. Creating category 4 model with backbone {args.fusion_backbone4}...")
        model_2d_global = ProtoECGNet2D(
            num_classes=num_classes4,
            single_class_prototype_per_class=args.fusion_single_ppc4,
            joint_prototypes_per_border=args.fusion_joint_ppb4,
            proto_dim=args.fusion_proto_dim4,
            proto_time_len=32,
            backbone=args.fusion_backbone4,
            prototype_activation_function=args.prototype_activation_function,
            latent_space_type=args.latent_space_type,
            add_on_layers_type=args.add_on_layers_type,
            class_specific=args.class_specific,
            last_layer_connection_weight=args.last_layer_connection_weight,
            m=args.m,
            dropout=trial_params["dropout"],
            custom_groups=True,
            label_set="4",
            pretrained_weights=args.fusion_weights4
        )

        print(f"Loaded all three models for fusion {args.fusion_backbone3}...")
        for m in [model_1d, model_2d_partial, model_2d_global]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        model = FusionProtoClassifier(model_1d, model_2d_partial, model_2d_global, num_classes=71)
        print(f"Fusion classifier initialized. Getting dataloaders...")
        train_loader, val_loader, test_loader, class_weights = get_fusion_dataloaders(args, return_sample_ids=False)

        print(f"Got dataloaders. Starting fusion classifier training...")
        trainer = train_model(model, train_loader, val_loader, train_args, class_weights)

        print(f"Training complete. Beginning fusion classifier testing...")
    else: 
        trainer = train_model(model, train_loader, val_loader, train_args, class_wts, trainer)

    # Evaluate on validation set
    val_results = test_model(model, test_loader=val_loader, args=train_args, trainer=trainer)
    val_auc = val_results[0]['test_auc']  # This is actually validation AUC

    # Also evaluate on test set
    test_results = test_model(model, test_loader=test_loader, args=train_args, trainer=trainer)
    test_auc = test_results[0]['test_auc'] # we do not optimize test AUC; this is just calculated for logging purposes

    print(f"Trial {trial.number} - {trial_name}: Val AUC = {val_auc:.4f}, Test AUC = {test_auc:.4f}")
    return val_auc  # Optuna will maximize validation AUC

# Save Study Callback
def save_study_callback(study, trial):
    save_study(study, study_save_path)
    print(f"Study saved to {study_save_path} after trial {trial.number}")

# Load or Create Study
if os.path.exists(study_save_path):
    print("Loading existing Optuna study...")
    study = load_study(study_save_path)
else:
    print("Creating new Optuna study...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=args.epochs, reduction_factor=3))

# Run Optuna Optimization
try:
    # Count trials by state
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    unfinished_trials = find_unfinished_trials(study)

    print(f"Study Summary: {len(study.trials)} total trials")
    print(f"  - Completed trials: {len(completed_trials)}")
    print(f"  - Pruned trials: {len(pruned_trials)}")
    print(f"  - Unfinished trials: {len(unfinished_trials)}")

    # Restart unfinished trials
    if unfinished_trials:
        print(f"Restarting unfinished trials: {unfinished_trials}")
        for trial_num in unfinished_trials:
            print(f"Retrying Trial {trial_num}...")
            try:
                study.tell(trial_num, None)  # Mark as incomplete so it restarts
                study.optimize(objective, n_trials=1, callbacks=[save_study_callback])
            except Exception as e:
                print(f"Trial {trial_num} failed again: {e}")
    else:
        print("No unfinished trials. Running new trials.")
        study.optimize(objective, n_trials=args.n_trials, callbacks=[save_study_callback])
except Exception as e:
    print(f"Error during optimization: {e}")

save_study(study, study_save_path)
print(f"Study saved to {study_save_path} after {len(study.trials)} trials.")

# Print Final Study Results
best_trial = study.best_trial
print("\nBest Trial:")
print(f"  Trial Number: {best_trial.number}")
print(f"  AUC: {best_trial.value}")
print("  Best Params:", best_trial.params)
