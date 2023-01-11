
import numpy as np
import warnings
import os
import os.path
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from utils import *
warnings.filterwarnings("ignore")
import config.cfg as cfg
args=cfg.parse_args()



def inception_score(imgs,Evaluate_loader, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = Evaluate_loader

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)

class Sample_evaluate(Dataset):
    def __init__(self, args, path, transform=None):
        self.transform = transform
        self.data_dir = path
        self.images_names = os.listdir(self.data_dir)


    def get_imgs(self, img_path, transform=None):
        img = Image.open(img_path).convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if transform is not None:
            img = transform(img)
        return img

    def __getitem__(self, index):
        key = self.images_names[index]
        img_name = '%s/%s' % (self.data_dir, key)
        imgs = self.get_imgs(img_name, self.transform)
        return imgs

    def __len__(self):
        return len(self.images_names)

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)
def calculate_IS(args,path):

    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = Sample_evaluate(args,path,transform=transform)
    Evaluate_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.gener_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    IgnoreLabelDataset(dataset)

    print("Calculating Inception Score...")

    Is_Score=inception_score(IgnoreLabelDataset(dataset), Evaluate_loader, cuda=True, batch_size=args.gener_batch_size, resize=True, splits=10)
    return Is_Score

