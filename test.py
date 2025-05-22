import time
import torch
from torch.utils.data import DataLoader
import tqdm
import datetime
import os

from compressor.compressor import Compressor
from decompressor.diffusion_manager import DiffusionManager
from decompressor.unet_module import UnetModule

import gc
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageOnlyDataset(Dataset):
    """
    A dataset that loads only images from a directory (no labels).
    All images are assumed to be the same size.
    """
    def __init__(self, image_dir, transform=None, cache_size=0, max_images=None):
        """
        Args:
            image_dir (str): Path to directory containing images
            transform (callable, optional): Transform to apply to images
            cache_size (int): Number of images to cache in memory (0 for no caching)
            max_images (int): Limit number of images loaded from the folder
        """
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # Converts images to tensors [0-1]
        ])
        self.cache_size = cache_size
        self.cache = {}
        
        # Get all image paths but don't load them yet
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            
        # Sort for reproducibility
        self.image_paths.sort()
        
        # Only keep a limited number of images
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Check if image is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load image only when needed
        image_path = self.image_paths[idx]
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                if self.cache_size > 0 and len(self.cache) < self.cache_size:
                    self.cache[idx] = image
                return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, 256, 256)  # Placeholder on failure

    def clear_cache(self):
        self.cache.clear()
        gc.collect()


# Setup the dataset and dataloader
def get_coco_patches_loader(
    data_dir="D:\\ds_coco_patches", 
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    cache_size=0,
    max_images=None
):
    """
    Creates a DataLoader for loading images from the specified directory.
    """
    dataset = ImageOnlyDataset(data_dir, cache_size=cache_size, max_images=max_images)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader, dataset


# ------------------------ Example usage ------------------------

BATCH_SIZE = 16  # Adjust based on your GPU memory
NUM_WORKERS = 0  # 0 = no multiprocessing
MAX_IMAGES = 100  # Limit to first 3000 images

train_loader, train_dataset = get_coco_patches_loader(
    data_dir="D:\\ds_coco_patches",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    cache_size=0,
    max_images=MAX_IMAGES
)

# Move a batch to CUDA and measure its size
for images in train_loader:
    images = images.cuda()  # Move to CUDA
    print(f"Batch size: {images.size(0)}")
    print(f"Batch shape: {images.shape}")  # Should be [B, 3, H, W]
    
    # Calculate memory usage
    batch_memory = images.element_size() * images.nelement() / (1024 ** 2)
    print(f"Estimated memory usage of the batch on CUDA: {batch_memory:.2f} MB")
    break



# Initialize the compressor
compressor = Compressor()

# Generate a dummy tensor to simulate input data
dummy_tensor = torch.randn(64, 3, 256, 256).cuda()

# Pass the dummy tensor through the compressor
output_dict = compressor(dummy_tensor)

# Extract the shapes of the output tensors
output_shapes = [output.shape[1] for output in output_dict['output']]

# Initialize the UNet module with the extracted channel dimensions
unet_module = UnetModule(context_channels=output_shapes)

del dummy_tensor

model = DiffusionManager(
    encoder=compressor,
    u_net=unet_module,
)


NUM_WORKERS = 0 if torch.cuda.is_available() else 0

# Create optimizer for both compressor and UNet
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters()},
    {'params': model.u_net.parameters()}
], lr=1e-4)

# Training parameters
num_epochs = 5
log_interval = 50  # Log every 50 batches
save_interval = 1   # Save checkpoint every epoch

# Create directory for saving checkpoints
save_dir = os.path.join("checkpoints", 
                        f"cdc_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(save_dir, exist_ok=True)

# Training statistics
train_losses = []
prior_losses = []

# Move models to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Starting training on {device}")
print(f"Training for {num_epochs} epochs with batch size {BATCH_SIZE}")

# Training loop
for epoch in range(num_epochs):
    train_dataset.clear_cache()  # Clear cache at the start of each epoch
    epoch_losses = []
    epoch_prior_losses = []

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"len train_loader: {len(train_loader)}")
    
    # Create tqdm progress bar
    progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                             desc=f"Epoch {epoch+1}/{num_epochs}")
    
    model.train()  # Set model to training mode
    
    for batch_idx, images in progress_bar:
        images = images.to(device)  # Move images to GPU
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass through diffusion model
        total_loss, prior_loss = model(images)
        
        
        del images
        torch.cuda.empty_cache()
        gc.collect()
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Record losses
        epoch_losses.append(total_loss.mean().item())
        epoch_prior_losses.append(prior_loss.item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{epoch_losses[-1]:.4f}",
            'prior_loss': f"{epoch_prior_losses[-1]:.4f}"
        })
        
        # Log periodically
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print(f"\nBatch {batch_idx}/{len(train_loader)}, "
                  f"Loss: {sum(epoch_losses[-log_interval:]) / log_interval:.4f}, "
                  f"Prior Loss: {sum(epoch_prior_losses[-log_interval:]) / log_interval:.4f}")
            
        time.sleep(5)
    
    # Calculate average epoch loss
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_prior_loss = sum(epoch_prior_losses) / len(epoch_prior_losses)
    train_losses.append(avg_loss)
    prior_losses.append(avg_prior_loss)
    
    print(f"\nEpoch {epoch+1}/{num_epochs} completed, "
          f"Avg Loss: {avg_loss:.4f}, "
          f"Avg Prior Loss: {avg_prior_loss:.4f}")
    
    # Save checkpoints
    if (epoch + 1) % save_interval == 0:
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': model.encoder.state_dict(),
            'unet_state_dict': model.u_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

# Plot training curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Total Loss')
plt.plot(range(1, num_epochs + 1), prior_losses, label='Prior Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

print("Training completed!")