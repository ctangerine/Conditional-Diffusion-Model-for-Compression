import gc
import os
import glob
import time
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
        # start_time = time.time()
        # Check if image is in cache
        if idx in self.cache:
            # print(f"Cache hit for index {idx} (time: {time.time() - start_time:.4f}s)")
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
                # print(f"Loaded image {image_path} (time: {time.time() - start_time:.4f}s)")
                return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, 256, 256)  # Placeholder on failure

    def clear_cache(self):
        # start_time = time.time()
        self.cache.clear()
        gc.collect()
        # print(f"Cache cleared (time: {time.time() - start_time:.4f}s)")


# Setup the dataset and dataloader
def get_coco_patches_loader(
    data_dir="D:\\ds_coco_patches", 
    batch_size=64,
    pin_memory=True,
    num_workers=4 if torch.cuda.is_available() else 0,
    shuffle=True,
    cache_size=0,
    max_images=None
):
    dataset = ImageOnlyDataset(data_dir, cache_size=cache_size, max_images=max_images)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader, dataset

# if __name__ == "__main__":
#     # Cấu hình loader
#     BATCH_SIZE = 4
#     NUM_WORKERS = 0  # Để đo chính xác, tránh đa luồng
#     MAX_IMAGES = 10  # Giới hạn số ảnh test nhanh
    
#     train_loader, train_dataset = get_coco_patches_loader(
#         data_dir="D:\\ds_coco_patches",
#         batch_size=BATCH_SIZE,
#         pin_memory=True,
#         num_workers=NUM_WORKERS,
#         shuffle=False,
#         cache_size=0,
#         max_images=MAX_IMAGES
#     )
    
#     print("\n--- Đo thời gian __getitem__ của dataset ---")
#     total_getitem_time = 0
#     for idx in range(len(train_dataset)):
#         start = time.time()
#         _ = train_dataset[idx]  # Gọi __getitem__
#         elapsed = time.time() - start
#         print(f"[__getitem__] Index {idx}: {elapsed:.4f} giây")
#         total_getitem_time += elapsed
    
#     print(f"Tổng thời gian __getitem__ cho {len(train_dataset)} ảnh: {total_getitem_time:.4f} giây")
#     print(f"Thời gian trung bình __getitem__ trên mỗi ảnh: {total_getitem_time / len(train_dataset):.4f} giây")
    
#     print("\n--- Đo thời gian lấy batch từ DataLoader ---")
#     total_batch_time = 0
#     for batch_idx, batch in enumerate(train_loader):
#         start = time.time()
#         # batch đã được yield từ DataLoader, gồm __getitem__ + collate
#         elapsed = time.time() - start
#         print(f"[DataLoader] Batch {batch_idx}: {elapsed:.4f} giây")
#         total_batch_time += elapsed
#         if batch_idx >= 2:
#             break
    
#     print(f"Tổng thời gian lấy 3 batch từ DataLoader: {total_batch_time:.4f} giây")
#     print(f"Thời gian trung bình lấy batch: {total_batch_time / (batch_idx+1):.4f} giây")
