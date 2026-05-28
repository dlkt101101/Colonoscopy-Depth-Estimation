import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from src.utils import scale_invariant_loss, DepthDataset
from src.train import MODEL_TYPE, get_transform

# loading the fine-tuned model
def load_model(weights_path: str, device: torch.device):
    """
    Loads the fine-tuned model for inference from the saved weights.
    """
    model = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
    # load pre-trained weights
    model = model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

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
