import csv
import datetime
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from config.args_pointnet_part_seg_train import *
from datasets.shapenet_seg_loader import ShapeNetDataset
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
    custom_transforms.ArrayToTensor(),
])

valid_transform = custom_transforms.Compose([
    custom_transforms.ArrayToTensor(),
])

train_set = ShapeNetDataset(Path(args.data_path), n_pts=args.n_pts, transform=train_transform, train=True)
val_set = ShapeNetDataset(Path(args.data_path), n_pts=args.n_pts, transform=valid_transform, train=False)

print('{} samples found in train scenes'.format(len(train_set)))
print('{} samples found in valid scenes'.format(len(val_set)))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

print("3.Creating Model")
shutil.copyfile('../models/pointnet_part_seg.py', args.save_path / 'pointnet_part_seg.py')
pointnet_part_seg = models.PointNet_part_seg(cls_num=args.num_cls, part_num=args.num_parts, input_transform=True,
                                             feature_transform=True).to(device)
if args.pretrained:
    print('=> using pre-trained weights for PoseNet')
    weights = torch.load(args.pretrained)
    pointnet_part_seg.load_state_dict(weights['state_dict'], strict=False)
else:
    pointnet_part_seg.init_weights()

print("4. Setting Optimization Solver")
optimizer = torch.optim.Adam(pointnet_part_seg.parameters(), lr=args.lr, betas=(args.momentum, args.beta),
                             weight_decay=args.weight_decay)

exp_lr_scheduler_R = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

print("5. Start Tensorboard ")
# tensorboard --logdir=/path_to_log_dir/ --port 6006
training_writer = SummaryWriter(args.save_path)

print("6. Create csvfile to save log information")

with open(args.save_path / args.log_summary, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['Total Train Loss', 'Validation Part_seg Accuracy'])

with open(args.save_path / args.log_full, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(
        ['total_loss', 'cls_loss', 'seg_loss', 'trans_feat_loss', ' Classification Accuracy', 'Segmentation Accuracy'])

print("7. Start Training!")


def main():
    best_error = -1
    for epoch in range(args.epochs):
        start_time = time.time()
        losses, loss_names = train(pointnet_part_seg, optimizer)
        errors, error_names = validate(pointnet_part_seg)

        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error > best_error
        best_error = max(best_error, decisive_error)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': pointnet_part_seg.state_dict()
        }, args.save_path / 'pointnet_part_seg_{}.pth.tar'.format(epoch))

        if is_best:
            shutil.copyfile(args.save_path / 'pointnet_part_seg_{}.pth.tar'.format(epoch),
                            args.save_path / 'pointnet_part_seg_best.pth.tar')

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
            csv_writer.writerow([losses[0], losses[1], losses[2], losses[3], losses[4]])

        print("\n---- [Epoch {}/{}] ----".format(epoch, args.epochs))
        print(
            "Train---Total loss:{}, Cls loss:{}, Seg loss:{}, Trans_feat loss:{}, Cls_Accuracy:{}, Seg_Accuracy:{}".format(
                losses[0],
                losses[1],
                losses[2],
                losses[3],
                losses[4],
                losses[5]))
        print("Valid---Cls_Accuracy:{}, Seg_Accuracy:{}".format(errors[0], errors[1]))

        epoch_left = args.epochs - (epoch + 1)
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time))
        print("----ETA {}".format(time_left))


def train(pointnet_cls, optimizer):
    loss_names = ['total_loss', 'cls_loss', 'seg_loss', 'trans_feat_loss', 'cls_accuracy', 'seg_accuracy']
    losses = AverageMeter(i=len(loss_names), precision=4)

    pointnet_cls.train()
    # points:(B,3, num_pts) label:(B,1) seg:(B,num_pts) label_ont_hot(batch_size,16)
    for i, (points, label, seg, label_one_hot) in enumerate(train_loader):
        label = torch.squeeze(label)  # (B,)
        points = points.to(device).float()
        label = label.to(device).long()
        seg = seg.to(device).long()
        label_one_hot = label_one_hot.to(device).float()
        # out_cls:(B,args.num_cls) out_seg:(B,args.num_parts,n_pts)
        out_cls, out_seg, trans_feat = pointnet_part_seg(points, label_one_hot)
        loss_1 = F.cross_entropy(out_cls, label)
        loss_2 = F.cross_entropy(out_seg, seg)
        loss_3 = feature_transform_regularizer(trans_feat)

        # 只有输入对其损失   0.001
        loss = (1 - args.seg_loss_weight) * loss_1 + args.seg_loss_weight * loss_2 + args.trans_loss_weight * loss_3

        cls_pred_val = torch.argmax(out_cls, 1)  # （B，）
        cls_correct = torch.sum(cls_pred_val == label)

        seg_pred_val = torch.argmax(out_seg, 1)  # （B,n_pts）
        seg_correct = torch.sum(seg_pred_val == seg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update([loss.item(), loss_1.item(), loss_2.item(), loss_3.item(), cls_correct.item() / args.batch_size,
                       seg_correct.item() / (args.batch_size * args.n_pts)], args.batch_size)

    return losses.avg, loss_names


@torch.no_grad()
def validate(pointnet_cls):
    error_names = ['Cls_Accuracy', 'Seg_Accuracy']
    errors = AverageMeter(i=len(error_names), precision=4)

    pointnet_cls.eval()

    for i, (points, label, seg, label_one_hot) in enumerate(train_loader):
        label = torch.squeeze(label)
        points = points.to(device).float()
        label = label.to(device).long()
        seg = seg.to(device).long()
        label_one_hot = label_one_hot.to(device).float()
        out_cls, out_seg, trans_feat = pointnet_part_seg(points, label_one_hot)

        cls_pred_val = torch.argmax(out_cls, 1)
        cls_correct = torch.sum(cls_pred_val == label)

        seg_pred_val = torch.argmax(out_seg, 1)
        seg_correct = torch.sum(seg_pred_val == seg)

        errors.update([cls_correct.item() / args.batch_size, seg_correct.item() / (args.batch_size * args.n_pts)],
                      args.batch_size)

    return errors.avg, error_names


if __name__ == '__main__':
    main()
