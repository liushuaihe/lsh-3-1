import pickle
import numpy as np
import tarfile
import io
import os

output_dir = os.path.join(os.path.dirname(__file__), 'static', 'testdata')
os.makedirs(output_dir, exist_ok=True)

num_images = 200
images = np.random.randint(0, 256, (num_images, 3072), dtype=np.uint8)
labels = np.random.randint(0, 10, num_images, dtype=np.int32)

data_dict = {
    b'data': images,
    b'labels': labels.tolist(),
    b'filenames': [f'image_{i:05d}.png' for i in range(num_images)]
}

tar_buffer = io.BytesIO()
with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
    data_bytes = pickle.dumps(data_dict)
    info = tarfile.TarInfo(name='cifar-10-batches-py/data_batch_1')
    info.size = len(data_bytes)
    tar.addfile(info, io.BytesIO(data_bytes))

output_path = os.path.join(output_dir, 'cifar10_test.tar.gz')
with open(output_path, 'wb') as f:
    f.write(tar_buffer.getvalue())

print(f'测试数据集已生成: {output_path}')
print(f'图片数量: {num_images}')
print(f'文件大小: {os.path.getsize(output_path) / 1024:.1f} KB')
