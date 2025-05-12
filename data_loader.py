import os
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection # Vẫn cần dùng CocoDetection để đọc thông tin ảnh
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LOGGER import setup_logger  # Giả sử bạn có một module logger.py để ghi log

PATCH_SIZE = (256, 256)  # Kích thước ảnh crop
COCO_ROOT_DIR = r"D:\temp_dataset\coco\images\train2017.1\train2017"  # Thư mục chứa dữ liệu COCO
TRAIN_IMAGES_SUBDIR = ""  # Thư mục chứa ảnh gốc
train_annotations_path = r"D:\temp_dataset\coco\images\train2017.1\annotations_trainval2017\annotations\captions_train2017.json"  # Đường dẫn đến file annotations


# --- Lớp Dataset tùy chỉnh chỉ để lấy ảnh từ CocoDetection ---
class CocoImagesOnlyDataset(Dataset):
    def __init__(self, coco_root, ann_file, image_subdir, transform=None):
        self.image_dir = os.path.join(coco_root, image_subdir)
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        try:
            self.coco_detection_dataset = CocoDetection(root=self.image_dir, annFile=ann_file)
        except Exception as e:
            print(f"Error initializing CocoDetection: {e}. Make sure pycocotools is installed.")
            raise
        self.transform = transform
        self.ids = list(sorted(self.coco_detection_dataset.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        coco_idx = self.ids[idx]
        img_info = self.coco_detection_dataset.coco.loadImgs(coco_idx)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # print(f"Warning: File not found {img_path}. Skipping.")
            # Trả về None hoặc raise lỗi để bỏ qua ảnh này trong vòng lặp lưu
            raise FileNotFoundError(f"Could not open image (file not found): {img_path}")
        except Exception as e:
            # print(f"Warning: Could not open image {img_path}: {e}. Skipping.")
            raise IOError(f"Could not open image {img_path}: {e}")

        if self.transform:
            img = self.transform(img)
        return img

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
            i = random.randint(0, original_height - target_height)
            j = random.randint(0, original_width - target_width)
            return transforms.functional.crop(img, i, j, target_height, target_width)
        else:
            short_dim_resized = max(self.output_size)
            img_resized = transforms.functional.resize(img, short_dim_resized, antialias=True)
            return transforms.functional.center_crop(img_resized, self.output_size)

# --- CHỌN PHIÊN BẢN TRANSFORM BẠN MUỐN SỬ DỤNG ---
# (Chọn một trong các train_transforms_v1, v2, v3 đã định nghĩa trước đó)
# train_transforms_v1 = transforms.Compose([...])
# train_transforms_v2 = transforms.Compose([...])
train_transforms_v3 = transforms.Compose([
    CustomRandomCropIfLargeEnough(PATCH_SIZE),
    transforms.RandomHorizontalFlip(), # Có thể bỏ nếu chỉ muốn crop
    transforms.ToTensor(),
])
selected_train_transforms = train_transforms_v3
# --- KẾT THÚC PHẦN GIẢ SỬ ---


# --- Cấu hình cho việc lưu ảnh ---
OUTPUT_DIR = r"D:\ds_coco_patches"  # Thư mục để lưu ảnh đã crop
NUM_IMAGES_TO_SAVE = 10000       # Số lượng ảnh crop bạn muốn lưu (điều chỉnh nếu cần)
                                # Đặt thành len(coco_train_dataset) để lưu tất cả (có thể rất nhiều)

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Sẽ lưu ảnh vào thư mục: {OUTPUT_DIR}")

# Khởi tạo Dataset
print("Đang khởi tạo dataset...")
try:
    coco_train_dataset = CocoImagesOnlyDataset(
        coco_root=COCO_ROOT_DIR,
        ann_file=train_annotations_path,
        image_subdir=TRAIN_IMAGES_SUBDIR,
        transform=selected_train_transforms # Sử dụng transform đã chọn
    )
    print(f"Dataset được khởi tạo thành công. Tổng số ảnh gốc: {len(coco_train_dataset)}")
except Exception as e:
    print(f"Lỗi khi khởi tạo dataset: {e}")
    exit()


# Hàm chuyển tensor sang PIL Image
to_pil = transforms.ToPILImage()

# Lặp và lưu ảnh
saved_count = 0
# Giới hạn số lượng ảnh lấy từ dataset, không vượt quá tổng số ảnh có trong dataset
num_to_process = min(NUM_IMAGES_TO_SAVE, len(coco_train_dataset))

print(f"Bắt đầu xử lý và lưu {num_to_process} ảnh crop...")
for i in tqdm(range(num_to_process)):
    try:
        # Lấy ảnh đã transform từ dataset
        # Mỗi lần gọi __getitem__, phép crop ngẫu nhiên (nếu có trong transform) sẽ được áp dụng
        img_tensor = coco_train_dataset[i]

        if img_tensor is None: # Trường hợp __getitem__ trả về None do lỗi (ví dụ)
            # print(f"Bỏ qua ảnh index {i} do không tải được.")
            continue

        # Chuyển tensor (C, H, W) sang ảnh PIL (H, W, C) rồi lưu
        pil_image = to_pil(img_tensor)

        # Tạo tên file duy nhất
        # Sử dụng ID ảnh gốc từ COCO nếu muốn truy vết lại, hoặc chỉ số i đơn giản
        # img_id = coco_train_dataset.ids[i] # Lấy ID ảnh gốc
        # filename = f"coco_{img_id}_patch_{i:07d}.png"
        filename = f"cropped_coco_img_{i:07d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        pil_image.save(filepath)
        saved_count += 1

    except FileNotFoundError as e:
        # print(f"Lỗi không tìm thấy file cho ảnh index {i}: {e}. Bỏ qua.")
        pass # Bỏ qua nếu ảnh gốc không tìm thấy
    except IOError as e:
        # print(f"Lỗi IO khi xử lý ảnh index {i}: {e}. Bỏ qua.")
        pass # Bỏ qua nếu có lỗi đọc ảnh
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý ảnh index {i}: {e}. Bỏ qua.")
        pass # Bỏ qua các lỗi khác

print(f"\nHoàn thành! Đã lưu {saved_count} ảnh crop vào thư mục: {OUTPUT_DIR}")

if saved_count < num_to_process:
    print(f"Lưu ý: Số lượng ảnh dự kiến lưu là {num_to_process}, nhưng chỉ có {saved_count} ảnh được lưu thành công (có thể do lỗi đọc file ảnh gốc).")