from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import argparse


# 固定随机数
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
random_seed(666, True)


# 数据集
class DefectDataset(Dataset):
    def __init__(self, img_dir, mask_dir, df, fold, img_size=224, is_valid=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        if is_valid:
            self.df = df[lambda x: x.fold == fold].drop(['fold'], axis=1).reset_index(drop=True)
        else:
            self.df = df[lambda x: x.fold != fold].drop(['fold'], axis=1).reset_index(drop=True)
        self.is_valid = is_valid
        self.cutout = v2.Compose([v2.RandomErasing(p=0.5, scale=(0.0001, 0.0003), ratio=(1, 1), value=128)] * 8)
        self.resize = v2.Resize(size=(img_size, img_size), antialias=True)
        self.rotate = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(90),
        ])
        self.colorjitter = v2.ColorJitter(0.1, 0.1)
        self.to_tensor = v2.Compose([
            v2.ToImageTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.d_label = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.fname.iloc[idx])
        img = Image.open(img_path)
        mask_path = os.path.join(self.mask_dir, self.df.fname.iloc[idx])
        mask = Image.open(mask_path)
        label = self.df.labels.iloc[idx]
        if self.is_valid:
            img = self.resize(img)
            img = self.to_tensor(img)
        else:
            mask = self.cutout(mask)
            img, mask = self.resize(img, mask)
            img, mask = self.rotate(img, mask)
            if random.random() > 0.5:
                img = self.colorjitter(img)
            img = np.array(img)
            mask = np.array(mask)
            img[mask == 0] = 128
            img = Image.fromarray(img)
            img = self.to_tensor(img)
        return img, self.d_label[label]
    
def get_ds(fold, img_size, bs):
    df = pd.read_csv('./src/split_5_folds.csv')
    train_ds = DefectDataset('./data/train/imgs/', './data/train/masks/', df, fold, img_size=img_size, is_valid=False)
    valid_ds = DefectDataset('./data/train/imgs/', './data/train/masks/', df, fold, img_size=img_size, is_valid=True)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)
    return train_dl, valid_dl


# 模型
def get_model(data):
    model = models.vit_h_14('IMAGENET1K_SWAG_LINEAR_V1')
    model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=5, bias=True)
    class_weight = torch.tensor([0.1, 0.3, 0.2, 0.1, 0.3])
    learn = Learner(data, model, loss_func=CrossEntropyLossFlat(weight=class_weight), 
                    opt_func=Adam, metrics=[accuracy])
    learn.model.cuda()
    return learn


parser = argparse.ArgumentParser()
parser.add_argument("fold", type=int, help="fold")
args = parser.parse_args()
fold = args.fold

train_dl, valid_dl = get_ds(fold, 224, 12)
data = DataLoaders(train_dl, valid_dl)

model_name = 'vit_h_14'
print('Model {} with fold {}.'.format(model_name, fold))
learn = get_model(data)
print('Finding lr...')
lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
print('Found lr: {}'.format(lr_min))

epochs = 100
lr = lr_min
Path('./logs/{}'.format(model_name)).mkdir(parents=True, exist_ok=True)
Path('./models/{}'.format(model_name)).mkdir(parents=True, exist_ok=True)
log_cb = CSVLogger(fname='./logs/{}/history_{}_{}.csv'.format(model_name, fold, lr), append=False)
es_cb = EarlyStoppingCallback(monitor='valid_loss', patience=10)
sm_cb = SaveModelCallback (monitor='valid_loss', fname='./{}/model_{}_{}'.format(model_name, fold, lr))
learn.fit_one_cycle(epochs, lr, cbs=[log_cb, es_cb, sm_cb])