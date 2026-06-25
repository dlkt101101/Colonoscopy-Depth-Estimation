import torch
import mlflow
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from src.utils import scale_invariant_loss, DepthDataset
from src.train import get_transform, get_best_run, MODEL_TYPE

# loading the fine-tuned model
def load_best_model(weights_path: str, device: torch.device):
    """
    Loads the fine-tuned model for inference from the saved weights.
    """
    best_params = get_best_run()
    model_uri = f"runs:/{best_params['run_id']}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    model.eval()
    return model, best_params

# make inference
def run_inference(model, image: Image.Image, device: torch.device):
    """
    Runs inference on the test set using the fine-tuned model and returns a 
    depth map as a numpy array.
    """
    transform = get_transform(MODEL_TYPE)
    input_tensor = transform(np.array(image.convert("RGB"))).to(device)

    with torch.no_grad():
        depth_map = model(input_tensor)

    return depth_map.squeeze().cpu().numpy()

def evaluate_on_test_set(model, test_dataset, best_batch_size, device: torch.device):
    """
    Evaluates the fine-tuned model on the test set and computes the average scale-invariant loss.
    """
    test_loss = 0.
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            color  = batch['color'].to(device)
            depths = batch['depth_gt'].to(device)
            preds = model(color)
            loss  = scale_invariant_loss(preds, depths)
            test_loss += loss.item()

        if i == 0:
            visualization_batch = {
                'color': color.cpu(),
                'depth_gt': depths.cpu(),
                'model_depth_map': preds.cpu()
            }

    average_test_loss = test_loss / len(test_loader)
    return average_test_loss, visualization_batch
