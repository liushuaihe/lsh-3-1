import os
import io
import pickle
import tarfile
import zipfile
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

CIFAR10_LABEL_NAMES = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]

@dataclass
class DatasetInfo:
    name: str
    total_images: int
    num_classes: int
    label_names: List[str]
    class_counts: Dict[str, int]
    format: str

class CIFAR10Dataset:
    def __init__(self):
        self.images: List[np.ndarray] = []
        self.labels: List[int] = []
        self.label_names: List[str] = CIFAR10_LABEL_NAMES
        self.name: str = ""
        self.format: str = ""
        self._loaded: bool = False

    def load_from_file(self, file_content: bytes, filename: str) -> DatasetInfo:
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.tar.gz') or filename_lower.endswith('.tgz'):
            return self._load_from_tar(file_content)
        elif filename_lower.endswith('.zip'):
            return self._load_from_zip(file_content)
        elif filename_lower.endswith('.bin'):
            return self._load_from_binary(file_content, filename)
        else:
            raise ValueError(f"不支持的文件格式: {filename}")

    def _load_from_tar(self, file_content: bytes) -> DatasetInfo:
        self.images = []
        self.labels = []
        
        file_bytes = io.BytesIO(file_content)
        
        try:
            with tarfile.open(fileobj=file_bytes, mode='r:gz') as tar:
                members = tar.getmembers()
                
                pickle_files = [m for m in members if 'data_batch' in m.name or 'test_batch' in m.name]
                bin_files = [m for m in members if m.name.endswith('.bin')]
                
                if pickle_files:
                    self.format = 'python-pickle'
                    for member in sorted(pickle_files, key=lambda m: m.name):
                        f = tar.extractfile(member)
                        if f:
                            data = pickle.load(f, encoding='bytes')
                            self._parse_pickle_batch(data)
                elif bin_files:
                    self.format = 'binary'
                    for member in sorted(bin_files, key=lambda m: m.name):
                        f = tar.extractfile(member)
                        if f:
                            content = f.read()
                            self._parse_binary_batch(content)
                else:
                    raise ValueError("未找到 CIFAR-10 数据文件（data_batch_* 或 *.bin）")
                
                self._finish_loading()
                return self._get_info()
                
        except Exception as e:
            raise ValueError(f"解析 tar.gz 文件失败: {str(e)}")

    def _load_from_zip(self, file_content: bytes) -> DatasetInfo:
        self.images = []
        self.labels = []
        
        file_bytes = io.BytesIO(file_content)
        
        try:
            with zipfile.ZipFile(file_bytes, 'r') as zf:
                names = zf.namelist()
                
                pickle_files = [n for n in names if 'data_batch' in n or 'test_batch' in n]
                bin_files = [n for n in names if n.endswith('.bin')]
                
                if pickle_files:
                    self.format = 'python-pickle'
                    for name in sorted(pickle_files):
                        with zf.open(name) as f:
                            data = pickle.load(f, encoding='bytes')
                            self._parse_pickle_batch(data)
                elif bin_files:
                    self.format = 'binary'
                    for name in sorted(bin_files):
                        with zf.open(name) as f:
                            content = f.read()
                            self._parse_binary_batch(content)
                else:
                    raise ValueError("未找到 CIFAR-10 数据文件（data_batch_* 或 *.bin）")
                
                self._finish_loading()
                return self._get_info()
                
        except Exception as e:
            raise ValueError(f"解析 zip 文件失败: {str(e)}")

    def _load_from_binary(self, file_content: bytes, filename: str) -> DatasetInfo:
        self.images = []
        self.labels = []
        self.format = 'binary'
        self._parse_binary_batch(file_content)
        self._finish_loading()
        self.name = filename
        return self._get_info()

    def _parse_pickle_batch(self, data: Dict):
        if b'data' in data:
            images_data = data[b'data']
            labels_data = data[b'labels']
        else:
            images_data = data['data']
            labels_data = data['labels']
        
        num_images = images_data.shape[0]
        images_data = images_data.reshape(num_images, 3, 32, 32)
        images_data = images_data.transpose(0, 2, 3, 1)
        
        for i in range(num_images):
            self.images.append(images_data[i])
            self.labels.append(int(labels_data[i]))

    def _parse_binary_batch(self, content: bytes):
        label_size = 1
        image_size = 32 * 32 * 3
        record_size = label_size + image_size
        
        num_records = len(content) // record_size
        
        for i in range(num_records):
            offset = i * record_size
            label = content[offset]
            
            image_data = content[offset + 1:offset + record_size]
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_array = image_array.reshape(3, 32, 32)
            image_array = image_array.transpose(1, 2, 0)
            
            self.images.append(image_array)
            self.labels.append(int(label))

    def _finish_loading(self):
        self._loaded = True

    def _get_info(self) -> DatasetInfo:
        class_counts = {}
        for label in self.labels:
            label_name = self.label_names[label] if label < len(self.label_names) else str(label)
            class_counts[label_name] = class_counts.get(label_name, 0) + 1
        
        return DatasetInfo(
            name=self.name or "CIFAR-10 数据集",
            total_images=len(self.images),
            num_classes=len(set(self.labels)),
            label_names=self.label_names,
            class_counts=class_counts,
            format=self.format
        )

    def get_image(self, index: int) -> Tuple[np.ndarray, int, str]:
        if not self._loaded:
            raise ValueError("数据集尚未加载")
        if index < 0 or index >= len(self.images):
            raise IndexError(f"图片索引超出范围: {index}")
        
        image = self.images[index]
        label = self.labels[index]
        label_name = self.label_names[label] if label < len(self.label_names) else str(label)
        
        return image, label, label_name

    def get_images_paginated(self, page: int = 1, per_page: int = 50, 
                              label_filter: Optional[int] = None) -> Dict:
        if not self._loaded:
            raise ValueError("数据集尚未加载")
        
        if label_filter is not None:
            indices = [i for i, l in enumerate(self.labels) if l == label_filter]
        else:
            indices = list(range(len(self.images)))
        
        total = len(indices)
        total_pages = (total + per_page - 1) // per_page
        
        start = (page - 1) * per_page
        end = min(start + per_page, total)
        page_indices = indices[start:end]
        
        images_base64 = []
        for idx in page_indices:
            image_array = self.images[idx]
            pil_image = Image.fromarray(image_array)
            pil_image = pil_image.resize((64, 64), Image.NEAREST)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            import base64
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            images_base64.append({
                "index": idx,
                "image": img_b64,
                "label": self.labels[idx],
                "label_name": self.label_names[self.labels[idx]]
            })
        
        return {
            "images": images_base64,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages
        }

    def get_image_high_res(self, index: int) -> Dict:
        if not self._loaded:
            raise ValueError("数据集尚未加载")
        if index < 0 or index >= len(self.images):
            raise IndexError(f"图片索引超出范围: {index}")
        
        image_array = self.images[index]
        label = self.labels[index]
        label_name = self.label_names[label] if label < len(self.label_names) else str(label)
        
        pil_image = Image.fromarray(image_array)
        pil_image = pil_image.resize((256, 256), Image.NEAREST)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        import base64
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "image": img_b64,
            "label": label,
            "label_name": label_name,
            "index": index
        }

    def is_loaded(self) -> bool:
        return self._loaded

    def clear(self):
        self.images = []
        self.labels = []
        self._loaded = False
        self.name = ""
        self.format = ""
