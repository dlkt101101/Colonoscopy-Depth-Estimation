# import necessary libraries
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

    tuning_results = []
    # hyper-parameter tuning loop
    for batch_size, lr in HYPER_PARAMETER_SET:
        print(f"\nTraining with BATCH_SIZE={batch_size}, LR={lr}")

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

        training_losses   = []
        validation_losses = []
        best_val_loss     = float('inf')

        for epoch in range(MAX_EPOCHS):
            # Training
            model.train()
            train_loss = 0.
            for batch in train_loader:
                color  = batch['color'].to(device)
                depths = batch['depth_gt'].to(device)
                optimizer.zero_grad()
                preds = model(color)
                loss  = scale_invariant_loss(preds, depths)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            training_losses.append(avg_train_loss)

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
            validation_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.4f} "
                  f"Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{MODEL_SAVE_DIR}/midas_bs{batch_size}_lr{lr}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved: {save_path}")

            early_stopping(avg_val_loss)
            if early_stopping.stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        tuning_results.append({
            'batch_size': batch_size,
            'lr': lr,
            'epochs_completed': len(training_losses),
            'training_loss': training_losses,
            'validation_loss': validation_losses,
            'best_validation_loss': best_val_loss
        })

    # Best hyperparameters
    sorted_results  = sorted(tuning_results, key=lambda x: x['best_validation_loss'])
    best_params     = sorted_results[0]
    best_lr         = best_params['lr']
    best_batch_size = best_params['batch_size']

    print(f"\nBest Config — LR: {best_lr}, Batch Size: {best_batch_size}, "
          f"Val Loss: {best_params['best_validation_loss']:.4f}")

    # Save tuning results
    pd.DataFrame(tuning_results).to_excel("tuning_results.xlsx", index=False)

    return best_params, test_dataset, transform, device


# if __name__ == "__main__":
#     run_training()