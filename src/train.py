# import necessary libraries
from copy import copy

import mlflow
import itertools
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from src.utils import *


# define constants
IMAGE_PATH        = "data/cecum_t1_a/color"
IMAGE_DEPTH_PATH  = "data/cecum_t1_a/depth"
MODEL_TYPE        = "DPT_Hybrid"
MAX_EPOCHS        = 10
BATCH_SIZE_LIST   = [4, 8, 12]
LEARNING_RATE_LIST = [1e-5, 1e-4, 1e-3]
HYPER_PARAMETER_SET = list(itertools.product(BATCH_SIZE_LIST, LEARNING_RATE_LIST))
MODEL_SAVE_DIR    = "models"
EXPERIMENT_NAME   = "Depth_Estimation_Experiment"

# set an experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# training functions
def get_transform(model_type):
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return transform

def run_training():
    transform = get_transform(MODEL_TYPE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare the data
    pairs = get_image_pairs(IMAGE_PATH, IMAGE_DEPTH_PATH)
    splits = train_val_test_split(pairs)

    train_dataset = DepthDataset(splits['train'], transform=transform)
    val_dataset = DepthDataset(splits['val'], transform=transform)
    test_dataset = DepthDataset(splits['test'], transform=transform)


    # hyper-parameter tuning loop
    for batch_size, lr in HYPER_PARAMETER_SET:
        with mlflow.start_run(run_name=f"bs{batch_size}_lr{lr}"):
            mlflow.log_param("model_type", MODEL_TYPE)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("max_epochs", MAX_EPOCHS)
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("loss_function", "scale_invariant_loss")
            mlflow.log_param("decoder_unfrozen", "scratch")
            mlflow.log_param("encoder_frozen", True)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

            model = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.scratch.parameters():
                param.requires_grad = True
            model.to(device)

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr
            )
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5, min_lr=1e-6
            )
            early_stopping = EarlyStopping(patience=5, min_delta=0.001)

            epochs_completed = 0
            best_val_loss = float('inf')

            for epoch in range(MAX_EPOCHS):
                # Training
                model.train()
                train_loss = 0.
                for batch in train_loader:
                    color = batch['color'].to(device)
                    depths = batch['depth_gt'].to(device)
                    optimizer.zero_grad()
                    preds = model(color)
                    loss = scale_invariant_loss(preds, depths)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # Validation
                model.eval()
                val_loss = 0.
                with torch.no_grad():
                    for batch in val_loader:
                        color  = batch['color'].to(device)
                        depths = batch['depth_gt'].to(device)
                        preds  = model(color)
                        val_loss += scale_invariant_loss(preds, depths).item()

                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                epochs_completed = epoch + 1

                # log mlflow metrics
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())

                early_stopping(avg_val_loss)
                if early_stopping.stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # log best metrics
            mlflow.log_metric("best_validation_loss", best_val_loss)
            mlflow.log_metric("epochs_completed", epochs_completed)
            mlflow.pytorch.log_model(model, "model")
    
    return get_best_run(), test_dataset

def get_best_run():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(order_by=["metrics.best_validation_loss ASC"], experiment_ids=[experiment.experiment_id])
    best_run = runs[0]

    # best parameters
    best_params = {
        'run_id': best_run.info.run_id,
        'batch_size': int(best_run.data.params["batch_size"]),
        'lr': float(best_run.data.params["learning_rate"]),
        'best_validation_loss': best_run.data.metrics["best_validation_loss"],
        'epochs_completed': int(best_run.data.metrics["epochs_completed"])
    }
    print(f"\nBest Config — LR: {best_params['lr']}, "
          f"Batch Size: {best_params['batch_size']}, "
          f"Val Loss: {best_params['best_validation_loss']:.4f}")

    return best_params

if __name__ == "__main__":
    run_training()