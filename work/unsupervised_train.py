'''
本脚本用于无监督训练， 获得预训练权重

'''
from unsupervised_by_contrast import ResNetSimCLR
from gaussian_blur import GaussianBlur

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchvision.transforms import transforms

import glob
import os
import argparse
import datetime
import numpy as np
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--weight', default="", type=str)
parser.add_argument('--temperature', default=0.07, type=float)


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, data_path1, data_path2):
        self.pretrain_dir_path1 = data_path1
        self.pretrain_dir_path2 = data_path2
        self.data = []
        for i in os.listdir(self.pretrain_dir_path1):
            path = os.path.join(self.pretrain_dir_path1, i)
            self.data.append(path)
        for j in os.listdir(self.pretrain_dir_path2):
            path = os.path.join(self.pretrain_dir_path2, j)
            self.data.append(path)
        self.transform = get_simclr_pipeline_transform(224)

    def __getitem__(self, item):
        img = Image.open(self.data[item])
        return [self.transform(img) for i in range(2)]

    def __len__(self):
        return len(self.data)


def main():
    args = parser.parse_args()

    # model
    model = ResNetSimCLR(2)
    model = model.cuda()

    # weight
    if args.weight:
        pth = torch.load(args.weight)
        model.load_state_dict(pth)
        print('weight {} loaded !'.format(args.weight))

    # data
    train_data = Mydataset('/home/std3/yyt/experiment2/dataset/100%/train/neg',
                           '/home/std3/yyt/experiment2/dataset/100%/train/pos')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )

    # criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    with open("contrastive_loss300(0.5).txt", "w") as f:
        for epoch in range(args.epochs):
            # train
            print("==== TRAINING ==========================================")
            start_time = datetime.datetime.now()
            model.train()
            epoch_loss = 0
            for i, images in enumerate(train_loader):

                images = torch.cat(images, dim=0)
                images = images.cuda()  # 2*B, C=3, 256, 256
                features = model(images)
                logits, labels = info_nce_loss(features, args)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            end_time = datetime.datetime.now()
            print("TRAINING EPOCH:::  {}/{} AVGLOSS::: {}  TIME COST::: {} s".format(epoch + 1, args.epochs,
                                                                                     epoch_loss / i,
                                                                                     (end_time - start_time).seconds))
            f.write("TRAINING EPOCH:::  {} AVGLOSS::: {}  TIME COST::: {} s".format(epoch + 1, epoch_loss / i,
                                                                                    (end_time - start_time).seconds))
            f.write('\n')
            f.flush()

            if epoch == 299:
                pth = model.state_dict()
                torch.save(pth, '/home/std3/yyt/experiment2/model/model300(0.5).pth')


def get_simclr_pipeline_transform(size, s=0.5):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          transforms.ToTensor()])
    return data_transforms


def info_nce_loss(features, args):
    labels = torch.cat([torch.arange(args.batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    logits = logits / args.temperature
    return logits, labels


if __name__ == '__main__':
    main()
