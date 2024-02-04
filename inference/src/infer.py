import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision.models as models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm


# 本次比赛用到的模型
model_names = [
    'convnext_large',
    'convnext_base',
    'maxvit_t',
    'vit_h_14',
]

d_info = {
    'convnext_large': {
        'weights': 'IMAGENET1K_V1',
        'change_head': lambda _: exec(
            "_.classifier[2] = nn.Linear(in_features=_.classifier[2].in_features, out_features=5, bias=True)"
        ),
        'img_size': 224,
        'bs': 32,
        'preds_weight': 0.3,
    },
    'convnext_base': {
        'weights': 'IMAGENET1K_V1',
        'change_head': lambda _: exec(
            "_.classifier[2] = nn.Linear(in_features=_.classifier[2].in_features, out_features=5, bias=True)"
        ),
        'img_size': 224,
        'bs': 64,
        'preds_weight': 0.3,
    },
    'maxvit_t': {
        'weights': 'IMAGENET1K_V1',
        'change_head': lambda _: exec(
            "_.classifier[5] = nn.Linear(in_features=_.classifier[5].in_features, out_features=5, bias=True)"
        ),
        'img_size': 224,
        'bs': 64,
        'preds_weight': 0.3,
    },
    'vit_h_14': {
        'weights': 'IMAGENET1K_SWAG_LINEAR_V1',
        'change_head': lambda _: exec(
            "_.heads.head = nn.Linear(in_features=_.heads.head.in_features, out_features=5, bias=True)"
        ),
        'img_size': 224,
        'bs': 12,
        'preds_weight': 0.1,
    },
}


def get_model(model_name, device):
    
    model = getattr(models, model_name)(weights=d_info[model_name]['weights'])
    d_info[model_name]['change_head'](model)
    model.to(device)
    return model


class TestDefectDataset(Dataset):
    
    def __init__(self, img_dir, df, img_size=224, flip_mode=0):
        self.img_dir = Path(img_dir)
        self.df = df
        self.resize = v2.Resize(size=(img_size, img_size), antialias=True)
        self.flip1 = v2.RandomHorizontalFlip(p=1)
        self.flip2 = v2.RandomVerticalFlip(p=1)
        self.to_tensor = v2.Compose([
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.flip_mode = flip_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = Path(self.df.fileName.iloc[idx])
        img_path = self.img_dir / '{}.png'.format(fn.stem)
        img = Image.open(img_path)
        if self.flip_mode == 0:
            transforms = v2.Compose([
                self.resize, 
                self.to_tensor, 
            ])
        elif self.flip_mode == 1:
            transforms = v2.Compose([
                self.resize, 
                self.flip1, 
                self.to_tensor, 
            ])
        elif self.flip_mode == 2:
            transforms = v2.Compose([
                self.resize, 
                self.flip2, 
                self.to_tensor, 
            ])
        elif self.flip_mode == 3:
            transforms = v2.Compose([
                self.resize, 
                self.flip1, 
                self.flip2, 
                self.to_tensor, 
            ])
        return transforms(img), 0
    
    
def predict_dl(model, dl, total, bs):
    preds = np.zeros((total, 5), dtype=np.float32)
    with torch.no_grad():
        for i, (batch, _) in enumerate(tqdm(dl)):
            preds[(i*bs):((i+1)*bs), :] = nn.Softmax(dim=1)(model(batch.to(device))).cpu().numpy()
    return preds


device = torch.device('cuda:0')
df = pd.read_csv('./data/test_B/提交样例.csv')
d_label = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}

d_preds = {}

for model_name in tqdm(model_names):
    model = get_model(model_name, device)
    model.eval()
    bs = d_info[model_name]['bs']

    all_preds = []
    for fn in tqdm(Path('./models/{}/'.format(model_name)).glob('*.pth')):
        state_dict = torch.load(fn, map_location=device)
        model.load_state_dict(state_dict)
        preds_TTA = []
        for i in tqdm(range(4)):
            test_ds = TestDefectDataset('./data/test_B/imgs/', df, flip_mode=i)
            test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False)
            preds_TTA.append(predict_dl(model, test_dl, len(test_ds), bs))
        preds = np.stack(preds_TTA, axis=0).mean(0)
        all_preds.append(preds)
        
    d_preds[model_name] = np.stack(all_preds, axis=0)
    
final_preds = np.zeros((len(test_ds), 5), dtype=np.float32)
for model_name in model_names:
    final_preds += d_info[model_name]['preds_weight'] * d_preds[model_name].mean(0)
    
df['defectType'] = [d_label[pred] for pred in final_preds.argmax(1)]
df.to_csv('./outputs/preds_testB.csv', index=None)
