import argparse
import os
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Lambda
from typing import Literal
from data import LoveDADataset, CityScapesDataset, CamVidDataset, PascalVOCDataset, EndoVisDataset
from stratifiers.kfold import KFoldWrapper
from stratifiers.wdes import WDESKFold
from stratifiers.ips import IPSKFold
import random

def get_dataset(name: str, path: str):
    annotation_transform = Lambda(lambda x: torch.as_tensor(np.expand_dims(np.array(x), 0), dtype=torch.int64))
    common_args = {'split': 'train', 'image_transform': ToTensor(), 'annotation_transform': annotation_transform}
    if name == 'cityscapes':
        return CityScapesDataset(path, **common_args)
    if name == 'loveda':
        return LoveDADataset(path, **common_args)
    if name == 'camvid':
        return CamVidDataset(path, **common_args)
    if name == 'pascalvoc':
        return PascalVOCDataset(path, **common_args)
    if name == 'endovis':
        return EndoVisDataset(path, **common_args)
    raise ValueError('Unsupported dataset {}'.format(name))

def get_model(name: str, num_channels: int, num_classes: int):
    if name == 'unet':
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=num_channels,
                        classes=num_classes)
    raise ValueError('Unsupported model {}'.format(name))

def get_stratifier(method: Literal['random', 'ips', 'wdes'], n_splits):
    if method == 'random':
        return KFoldWrapper(n_splits=n_splits)
    elif method == 'ips':
        return IPSKFold(n_splits=n_splits)
    elif method == 'wdes':
        return WDESKFold(n_splits=n_splits)

def log_results(dataset, stratify, n_splits, fold, iou, f1, acc):
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"{dataset}.txt")
    with open(report_path, "a") as f:
        f.write(f"{stratify} {n_splits} {fold} IoU: {iou:.4f} F1: {f1:.4f} Acc: {acc:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Train segmentation models with stratification')
    parser.add_argument('--stratify', '-s', choices=['random', 'ips', 'wdes'], required=True)
    parser.add_argument('--model', '-m', choices=['unet'], required=True)
    parser.add_argument('--epochs', '-e', type=int, required=True)
    parser.add_argument('--learning-rate', '-lr', type=float, required=True)
    parser.add_argument('--batch-size', '-bs', type=int, required=True)
    parser.add_argument('--dataset', '-d',
                        choices=['cityscapes', 'camvid', 'pascalvoc', 'loveda', 'endovis'], required=True)
    parser.add_argument('--path', '-p', required=True)
    parser.add_argument('--n_splits', required=True, type=int)
    parser.add_argument('--fold', '-f', nargs='+', type=int)
    args = parser.parse_args()

    assert all(f < args.n_splits for f in args.fold), "Fold number should be less than number of splits"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Running on {}'.format(device))

    dataset = get_dataset(args.dataset, args.path)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    stratifier = get_stratifier(args.stratify, args.n_splits)

    for i, (train_idx, test_idx) in enumerate(stratifier.split(dataset)):
        if i not in args.fold:
            continue

        model = get_model(args.model, dataset.num_channels, dataset.num_classes).to(device)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        if args.dataset == 'pascalvoc':
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = torch.nn.CrossEntropyLoss()

        for e in range(args.epochs):
            model.train()
            for images, annotations in train_loader:
                images, annotations = images.to(device), annotations.to(device)
                optimizer.zero_grad()
                predictions = model(images)
                if args.dataset == 'pascalvoc':
                    loss = criterion(predictions, annotations.squeeze())
                else:
                    loss = criterion(predictions, annotations)
                loss.backward()
                optimizer.step()

        model.eval()
        stats = []
        with torch.no_grad():
            for images, annotations in test_loader:
                images, annotations = images.to(device), annotations.to(device)
                predictions = model(images).argmax(dim=1)
                annotations = annotations.squeeze(dim=1)
                stats.append(
                    smp.metrics.get_stats(predictions, annotations, mode='multiclass', num_classes=dataset.num_classes)
                )

        tp = torch.cat([x[0] for x in stats])
        fp = torch.cat([x[1] for x in stats])
        fn = torch.cat([x[2] for x in stats])
        tn = torch.cat([x[3] for x in stats])

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro").item()
        acc = smp.metrics.balanced_accuracy(tp, fp, fn, tn, reduction="macro").item()

        log_results(args.dataset, args.stratify, args.n_splits, i, iou, f1, acc)

if __name__ == '__main__':
    main()
