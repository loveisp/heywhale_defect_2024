import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from multiprocessing import Pool


def get_image(full_fn):
    data = pd.read_csv(full_fn)
    
    xmin = data.X.min()
    xmax = data.X.max()
    ymin = data.Y.min()
    ymax = data.Y.max()
    vmin = data.Value.min()
    vmax = data.Value.max()
    
    img_size = (ymax - ymin + 1, xmax - xmin + 1)
    img = np.ones(img_size, dtype=np.uint8) * 128  # 背景设为中间值，其实无所谓，因为可以用mask给去掉
    mask = np.zeros(img_size, dtype=np.uint8)
    for i, row in data.iterrows():
        x, y, val = row.X, row.Y, row.Value
        img[int(y-ymin), int(x-xmin)] = np.uint8((val - vmin) / (vmax - vmin + 1e-5) * 256)
        mask[int(y-ymin), int(x-xmin)] = 1
    return img, mask


img_path = Path('./data/test_B/imgs')
img_path.mkdir(parents=True, exist_ok=True)
mask_path = Path('./data/test_B/masks')
mask_path.mkdir(parents=True, exist_ok=True)
csv_path = Path('./data/test_B/csv文件/')

def _foo(fn):
    img, mask = get_image(csv_path / fn)
    Image.fromarray(img).save(img_path / '{}.png'.format(fn.split('.')[0]))
    Image.fromarray(mask).save(mask_path / '{}.png'.format(fn.split('.')[0]))


df = pd.read_csv('./data/test_B/提交样例.csv')
processes = 16
with Pool(processes) as p:
    list(tqdm(p.imap(_foo, df.fileName.values), total=len(df)))