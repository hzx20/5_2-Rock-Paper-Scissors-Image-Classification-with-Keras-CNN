import os
import urllib.request
import zipfile
import time

DATA_DIR = 'rps_data'
TRAIN_ZIP = os.path.join(DATA_DIR, 'rps.zip')
TEST_ZIP = os.path.join(DATA_DIR, 'rps-test-set.zip')
TRAIN_DIR = os.path.join(DATA_DIR, 'rps')
TEST_DIR = os.path.join(DATA_DIR, 'rps-test-set')

os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_URL = 'https://storage.googleapis.com/learning-datasets/rps.zip'
TEST_URL = 'https://storage.googleapis.com/learning-datasets/rps-test-set.zip'


def download_with_progress(url, filepath):
    if os.path.exists(filepath):
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                names = zf.namelist()
                print(f'{os.path.basename(filepath)} 已存在且有效 ({len(names)} 个文件)')
                return
        except Exception:
            print(f'{os.path.basename(filepath)} 已损坏，重新下载...')
            os.remove(filepath)

    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024 if total_size > 0 else 0
        if block_num % 10 == 0 or downloaded >= total_size:
            print(f'\r  进度: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='', flush=True)

    print(f'正在下载: {os.path.basename(url)}')
    start = time.time()
    urllib.request.urlretrieve(url, filepath, reporthook=hook)
    elapsed = time.time() - start
    size = os.path.getsize(filepath) / 1024 / 1024
    print(f'\r  完成! 大小: {size:.2f} MB, 耗时: {elapsed:.1f} 秒')


def extract_zip(zip_path, extract_to):
    if os.path.exists(extract_to):
        print(f'{os.path.basename(extract_to)} 已解压')
        return
    print(f'正在解压: {os.path.basename(zip_path)}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        total = len(names)
        for i, name in enumerate(names):
            zf.extract(name, DATA_DIR)
            if (i + 1) % 100 == 0 or i + 1 == total:
                print(f'\r  进度: {i + 1}/{total}', end='', flush=True)
        print()
    print(f'  解压完成!')


print('=' * 60)
print('石头剪刀布数据集 - 下载与解压')
print('=' * 60)

download_with_progress(TRAIN_URL, TRAIN_ZIP)
extract_zip(TRAIN_ZIP, TRAIN_DIR)

download_with_progress(TEST_URL, TEST_ZIP)
extract_zip(TEST_ZIP, TEST_DIR)

print()
print('数据集验证:')
print('-' * 60)
for name, d in [('训练集', TRAIN_DIR), ('测试集', TEST_DIR)]:
    if os.path.exists(d):
        cats = sorted([c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))])
        print(f'{name}:')
        total = 0
        for c in cats:
            count = len(os.listdir(os.path.join(d, c)))
            print(f'  {c}: {count} 张图片')
            total += count
        print(f'  总计: {total} 张图片')
    else:
        print(f'{name}: 目录不存在!')

print()
print('数据集准备完成!')
