# install the required libraries
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# initializing some constants
RANDOM_SEED = 123
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# define the classes
class DepthDataset(Dataset):
  """
  Class creates a PyTorch dataset.
  """
  def __init__(self, pairs: list, depth_size = (384, 480), transform=None):
    self.transform = transform
    self.pairs = pairs
    self.depth_size = depth_size

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    pair = self.pairs[idx]
    color_image = np.array(Image.open(pair['color']).convert('RGB'))
    depth_image = Image.open(pair['depth'])
    depth_image = depth_image.resize(
        (self.depth_size[1], self.depth_size[0]), resample=Image.NEAREST
    )

    depth_image = np.array(depth_image)

    # Use MiDaS transformation for the images
    if self.transform:
      color_image = self.transform(color_image).squeeze(0)

    return {
        'color': color_image,
        'depth_gt': torch.tensor(depth_image, dtype=torch.float32)
    }

class EarlyStopping():
  """
  Class performs early stopping on model training and validation.
  """
  def __init__(self, patience=5, min_delta=0.001):
    self.patience = patience
    self.min_delta = min_delta
    self.stop = False
    self.best_loss = float('inf')
    self.counter = 0

  def __call__(self, val_loss):
    if val_loss < self.best_loss:
      self.counter  = 0
      self.best_loss = val_loss
    else:
      self.counter += 1
      if self.patience >= self.counter:
        self.stop =True

# define helper fuinctions
def resize_predictions(predictions, ground_truths):
  """
  Resizes the model predictions to ground truths size. Result is used for calculating
  loss function.
  """
  resized = torch.nn.functional.interpolate(predictions.unsqueeze(1),
                                            mode='bicubic',
                                            size = ground_truths.shape[-2:],
                                            align_corners=False).squeeze(1)
  return resized


def scale_invariant_loss(prediction, target):
  """
  A scale-invariant loss function specifically for MiDaS.
  """
  mask = target > 0 # filter out all invalid pixels
  prediction = prediction[mask]
  target = target[mask]
  M = mask.sum().float()

  # normalize the pixel values for target and prediction
  # adding a small value to standard deviation to prevent division by 0
  prediction_norm = (prediction - prediction.mean()) / (prediction.std() + 1e-6)
  target_norm = (target - target.mean()) / (target.std() + 1e-6)

  # Loss function
  loss = 1.0 / (2. * M) * torch.sum(torch.abs(prediction_norm - target_norm))
  return loss

# create train, validation and test split for the dataset
def train_val_test_split(pairs: list, random_state = RANDOM_SEED, train_ratio=TRAIN_RATIO,
                         val_ratio = VAL_RATIO, test_ratio=TEST_RATIO):
  """
  Function creates training, validation and test split for image pairs.

  Returns a dictionary for such split.
  """
  train_pairs, temp_pairs = train_test_split(pairs, test_size= test_ratio + val_ratio, random_state = RANDOM_SEED)

  val_pairs, test_pairs = train_test_split(temp_pairs, test_size = 0.5, random_state = RANDOM_SEED)

  splits = {
      'train': train_pairs,
      'val': val_pairs,
      'test': test_pairs
  }
  return splits

def get_image_pairs(img_folder: str, depth_folder: str):
  """
  The function gets a pair of the image and its depth map image and returns a
  list of dictionaries.
  """
  # Pre-emptive checks
  if os.path.exists(img_folder) and os.path.exists(depth_folder):
    img_files   = glob.glob(os.path.join(img_folder, '*.png'))
    depth_files = glob.glob(os.path.join(depth_folder, '*.tiff'))
    img_folder_count   = len(img_files)
    depth_folder_count = len(depth_files)

  else:
    print('One or both files do not exist.')

  if img_folder_count != depth_folder_count:
    print('Mismatch in number of images found in files. {} missing'.format(abs(img_folder_count - depth_folder_count)))

  # get the image pairs
  depth_dict = {}
  for path in depth_files:
    filename = os.path.basename(path)
    _match = re.match(r'^(\d{4})_', filename)
    if _match:
      depth_dict[_match.group(1)] = path

  pairs = list()
  unmatched = list()

  for color_path in sorted(img_files):
    filename = os.path.basename(color_path)
    image_match = re.match(r'^(\d+)_', filename)
    if image_match:
      image_prefix = image_match.group(1).zfill(4)
      if image_prefix in depth_dict:
        pairs.append({
            'color': color_path,
            'depth': depth_dict[image_prefix]
        })
      else:
        unmatched.append(filename)

  if unmatched:
    print('No depth match found for {} color image (s): {}'.format(len(unmatched), unmatched))

  print('Successfully paired {} images'.format(len(pairs)))
  return pairs

def resize_predictions(predictions, ground_truths):
  """
  Resizes the model predictions to ground truths size. Result is used for calculating
  loss function.
  """
  resized = torch.nn.functional.interpolate(predictions.unsqueeze(1),
                                            mode='bicubic',
                                            size = ground_truths.shape[-2:],
                                            align_corners=False).squeeze(1)
  return resized

def visualize_predictions(combined_batch, best_batch_size, best_lr):
  """
  Visualize the predictions of the base and tuned models alongside the ground truth depth maps and original color images.
  """
  row_labels = ['Colon', 'Ground Truth Depth Map',
                  'Base Model Depth Map', 'Tuned Model Depth Map']
  num_images = 4
  fig, axes = plt.subplots(nrows=4, ncols=num_images, figsize=(20, 16))
  for i in range(num_images):
    color_img = combined_batch['color'][i].permute(1, 2, 0).numpy()
    color_img = (color_img - color_img.min()) / (color_img.max() - color_img.min())

    axes [0, i].imshow(color_img)
    axes [1, i].imshow(combined_batch['depth_gt'][i].numpy())
    axes [2, i].imshow(combined_batch['base_depth_pred'][i].numpy())
    axes [3, i].imshow(combined_batch['tuned_depth_pred'][i].numpy())

    for row in range(4):
      axes[row, i].axis('off')
    
    for row, label in enumerate(row_labels):
      for col in range(num_images):
        axes[row, col].set_title(f'{label}; Image {col+1}')
    
    plt.suptitle(f'Base vs Tuned Model | BS={best_batch_size} LR={best_lr}')
    plt.tight_layout()
    plt.show()