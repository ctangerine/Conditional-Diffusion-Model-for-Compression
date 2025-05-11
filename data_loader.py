import os
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection # Vẫn cần dùng CocoDetection để đọc thông tin ảnh
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LOGGER import setup_logger  # Giả sử bạn có một module logger.py để ghi log

COCO_ROOT_DIR = r"D:\temp_dataset\coco\images\train2017.1\train2017"  # Thư mục gốc chứa COCO
TRAIN_IMAGES_SUBDIR = ""     # Thư mục con chứa ảnh train2017
TRAIN_ANNOTATIONS_FILENAME = "captions_train2017.json" # File chú thích cho train2017

PATCH_SIZE = (256, 256)  # Kích thước patch mong muốn (height, width)
BATCH_SIZE = 64          # Kích thước batch cho DataLoader
NUM_WORKERS = 0         # Số luồng để tải dữ liệu

# --- Xây dựng đường dẫn đầy đủ ---
train_images_path = os.path.join(COCO_ROOT_DIR, TRAIN_IMAGES_SUBDIR)
train_annotations_path = os.path.join(r"D:\temp_dataset\coco\images\train2017.1\annotations_trainval2017\annotations", TRAIN_ANNOTATIONS_FILENAME)


# --- Lớp Dataset tùy chỉnh chỉ để lấy ảnh từ CocoDetection ---
class CocoImagesOnlyDataset(Dataset):
    def __init__(self, coco_root, ann_file, image_subdir, transform=None):
        self.image_dir = os.path.join(coco_root, image_subdir)
        self.coco_detection_dataset = CocoDetection(root=self.image_dir, annFile=ann_file)
        self.transform = transform
        self.ids = list(sorted(self.coco_detection_dataset.ids)) # Lấy danh sách ID ảnh

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        coco_idx = self.ids[idx] # Lấy ID ảnh COCO thực sự

        img_info = self.coco_detection_dataset.coco.loadImgs(coco_idx)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file {img_path}. Sử dụng ảnh mặc định dogpicture.png.")
            img = Image.open("dogpicture.png").convert('RGB')
        except Exception as e:
            print(f"Lỗi không xác định khi mở file {img_path}: {e}. Sử dụng ảnh mặc định dogpicture.png.")
            img = Image.open("dogpicture.png").convert('RGB')


        if self.transform:
            img = self.transform(img)

        return img

train_transforms_v1 = transforms.Compose([
    transforms.RandomResizedCrop(PATCH_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)), # Thử scale lớn hơn
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Tùy chọn
])

RESIZE_LARGER_DIM = max(PATCH_SIZE) + 64 # Ví dụ: 256 + 64 = 320
train_transforms_v2 = transforms.Compose([
    transforms.Resize(RESIZE_LARGER_DIM), # Resize chiều ngắn hơn thành RESIZE_LARGER_DIM
    transforms.RandomCrop(PATCH_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Tùy chọn
])

# Cách 3: Transform tùy chỉnh để chỉ cắt nếu ảnh đủ lớn (giống logic Vimeo-90k hơn)
class CustomRandomCropIfLargeEnough:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img):
        original_width, original_height = img.size
        target_height, target_width = self.output_size

        if original_width >= target_width and original_height >= target_height:
            # Ảnh đủ lớn, thực hiện RandomCrop
            i = random.randint(0, original_height - target_height)
            j = random.randint(0, original_width - target_width)
            return transforms.functional.crop(img, i, j, target_height, target_width)
        else:
            # Ảnh quá nhỏ, resize lên rồi CenterCrop (hoặc bạn có thể chọn RandomCrop)
            # Việc resize có thể làm thay đổi tỷ lệ khung hình nếu không cẩn thận
            # Ở đây ta resize theo chiều ngắn hơn rồi CenterCrop
            short_dim_resized = max(self.output_size) # Resize chiều ngắn hơn của ảnh thành chiều lớn hơn của patch
            img_resized = transforms.functional.resize(img, short_dim_resized)
            return transforms.functional.center_crop(img_resized, self.output_size)

train_transforms_v3 = transforms.Compose([
    CustomRandomCropIfLargeEnough(PATCH_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


# --- CHỌN PHIÊN BẢN TRANSFORM BẠN MUỐN SỬ DỤNG ---
# selected_train_transforms = train_transforms_v1 # Hoặc v2, v3
# selected_train_transforms = train_transforms_v2
selected_train_transforms = train_transforms_v3


coco_train_dataset = CocoImagesOnlyDataset(
    coco_root=COCO_ROOT_DIR,
    ann_file=train_annotations_path,
    image_subdir=TRAIN_IMAGES_SUBDIR,
    transform=selected_train_transforms
)