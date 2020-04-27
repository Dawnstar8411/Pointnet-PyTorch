import csv
import datetime
import shutil
import time
import warnings

import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from config.args_pointnet_cls_train import *
from datasets.modelnet_cls_loader import ModelNetDataset
from utils import custom_transforms
from utils.loss_functions import feature_transform_regularizer
from utils.utils import AverageMeter, save_path_formatter

warnings.filterwarnings('ignore')
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda") if args.cuda else torch.device("cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("1. Path to save the output.")
save_path = Path(save_path_formatter(args))
args.save_path = 'checkpoints' / save_path
args.save_path.makedirs_p()
print("=> will save everything to {}".format(args.save_path))

print("2.Data Loading...")

train_transform = custom_transforms.Compose([
    custom_transforms.select_point_cloud(n_pts=args.n_pts),
    custom_transforms.rotate_point_cloud(),
    custom_transforms.jitter_point_cloud(sigma=0.01, clip=0.05),
    custom_transforms.ArrayToTensor(),
])

valid_transform = custom_transforms.Compose([
    custom_transforms.select_point_cloud(n_pts=args.n_pts),
    custom_transforms.ArrayToTensor(),
])

train_set = ModelNetDataset(Path(args.data_path), n_pts=args.n_pts, transform=train_transform, train=True)
val_set = ModelNetDataset(Path(args.data_path), n_pts=args.n_pts, transform=valid_transform, train=False)

print('{} samples found in train scenes'.format(len(train_set)))
print('{} samples found in valid scenes'.format(len(val_set)))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

print("3.Creating Model")
shutil.copyfile('../models/pointnet_cls.py', args.save_path / 'pointnet_cls.py')
pointnet_cls = models.PointNet_cls(num_cls=40, input_transform=True, feature_transform=True).to(device)
if args.pretrained:
    print('=> using pre-trained weights for PoseNet')
    weights = torch.load(args.pretrained)
    pointnet_cls.load_state_dict(weights['state_dict'], strict=False)
else:
    pointnet_cls.init_weights()

print("4. Setting Optimization Solver")
optimizer = torch.optim.Adam(pointnet_cls.parameters(), lr=args.lr, betas=(args.momentum, args.beta),
                             weight_decay=args.weight_decay)

exp_lr_scheduler_R = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

print("5. Start Tensorboard ")
# tensorboard --logdir=/path_to_log_dir/ --port 6006
training_writer = SummaryWriter(args.save_path)

print("6. Create csvfile to save log information")

with open(args.save_path / args.log_summary, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['Total Train Loss', 'Validation Classification Accuracy'])

with open(args.save_path / args.log_full, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(
        ['total_loss', 'cls_loss', 'trans_feat_loss', 'Train cls Accuracy' 'Validation cls Accuracy'])

print("7. Start Training!")


def main():
    best_error = -1
    for epoch in range(args.epochs):
        start_time = time.time()
        losses, loss_names = train(pointnet_cls, optimizer)
        errors, error_names = validate(pointnet_cls)

        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error > best_error
        best_error = max(best_error, decisive_error)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': pointnet_cls.state_dict()
        }, args.save_path / 'pointnet_cls_{}.pth.tar'.format(epoch))

        if is_best:
            shutil.copyfile(args.save_path / 'pointnet_cls_{}.pth.tar'.format(epoch),
                            args.save_path / 'pointnet_cls_best.pth.tar')

        for loss, name in zip(losses, loss_names):
            training_writer.add_scalar(name, loss, epoch)
            training_writer.flush()
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
            training_writer.flush()

        with open(args.save_path / args.log_summary, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([losses[0], decisive_error])

        with open(args.save_path / args.log_full, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([losses[0], losses[1], losses[2], losses[3], errors[0]])

        print("\n---- [Epoch {}/{}] ----".format(epoch, args.epochs))
        print(
            "Train---Total loss:{}, Cls loss:{}, Trans_feat loss:{}, Accuracy:{}".format(losses[0], losses[1],
                                                                                         losses[2], losses[3]))
        print("Valid---Accuracy:{}".format(errors[0]))

        epoch_left = args.epochs - (epoch + 1)
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time))
        print("----ETA {}".format(time_left))


def train(pointnet_cls, optimizer):
    loss_names = ['total_loss', 'cls_loss', 'trans_feat_loss', 'Train Accuracy']
    losses = AverageMeter(i=len(loss_names), precision=4)

    pointnet_cls.train()

    # points:(B,3,n_pts)  label:(B,1)
    for i, (points, label) in enumerate(train_loader):
        label = torch.squeeze(label) # (B,)
        points, label = points.to(device).float(), label.to(device).long()
        targets, trans_feat = pointnet_cls(points) # targets:(B, args.num_cls)
        pred_val = torch.argmax(targets, 1)  # (B,)
        correct = torch.sum(pred_val == label)
        loss_1 = F.cross_entropy(targets, label)
        loss_2 = feature_transform_regularizer(trans_feat)
        loss = loss_1 + 0.001 * loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update([loss.item(), loss_1.item(), loss_2.item(), correct.item() / args.batch_size], args.batch_size)

    return losses.avg, loss_names


@torch.no_grad()
def validate(pointnet_cls):
    error_names = ['Accuracy']
    errors = AverageMeter(i=len(error_names), precision=4)
    pointnet_cls.eval()
    # points:(B,3,n_pts)  label:(B,1)
    for i, (points, label) in enumerate(val_loader):
        label = torch.squeeze(label) # (B,)
        points, label = points.to(device).float(), label.to(device).long()
        targets, _ = pointnet_cls(points) # targets:(B, args.num_cls)

        pred_val = torch.argmax(targets, 1) # (B,)
        correct = torch.sum(pred_val == label)
        errors.update([correct.item() / args.batch_size], args.batch_size)

    return errors.avg, error_names


if __name__ == '__main__':
    main()
