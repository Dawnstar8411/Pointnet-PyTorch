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
from utils.utils import AverageMeter, save_path_formatter
from utils.loss_functions import feature_transform_regularizer

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

train_set = ShapeNetDataset(Path(args.data_path), npoints=args.n_pts, transform=train_transform, train=True)
val_set = ShapeNetDataset(Path(args.data_path), npoints=args.n_pts, transform=valid_transform, train=False)

print('{} samples found in train scenes'.format(len(train_set)))
print('{} samples found in valid scenes'.format(len(val_set)))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.wokers, pin_memory=True)

print("3.Creating Model")

pointnet_part_seg = models.PointNet_part_seg(cls_num=args.num_cls, part_num=args.num_parts, input_transform=True, feature_transform=True).to(device)
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
    csv_writer.writerow(['train_loss', 'validation_loss'])

with open(args.save_path / args.log_full, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['total_loss', 'nll_loss', 'trans_loss', 'trans_feat_loss' 'mAP'])

print("7. Start Training!")


def main():
    best_error = -1
    for epoch in range(args.epochs):
        start_time = time.time()
        losses, loss_names = train(pointnet_part_seg, optimizer)
        errors, error_names = validate(pointnet_part_seg)

        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': pointnet_part_seg.state_dict()
        }, args.save_path / 'pointnet_cls{}.pth.tar'.format(epoch))

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
            csv_writer.writerow([losses[0], losses[1], losses[3]])

        print("\n---- [Epoch {}/{}] ----".format(epoch, args.epochs))
        print("Train---Total loss:{}, Nll_loss:{}, Trans loss:{}, Trans_feat loss:{}".format(losses[0], losses[1],
                                                                                             losses[2], losses[3]))
        print("Valid---mAp:{}".format(errors[0]))

        epoch_left = args.epochs - (epoch + 1)
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time))
        print("----ETA {}".format(time_left))


def train(pointnet_cls, optimizer):
    loss_names = ['total_loss', 'nll_loss', 'trans_loss', 'trans_feat_loss']
    losses = AverageMeter(i=len(loss_names), precision=4)

    pointnet_cls.train()
    # points:(batch_size,3, num_pts) label:(batch_size,1) seg:(batch_size,num_pts) label_ont_hot(batch_size,16)
    for i, (points, label,seg,label_ont_hot) in enumerate(train_loader):
        targets, trans, trans_feat = pointnet_cls(points)
        loss_1 = F.cross_entropy(label, targets)
        loss_2 = feature_transform_regularizer(trans)
        loss_3 = feature_transform_regularizer(trans_feat)
        # 只有输入对其损失   0.001
        loss = loss_1 + loss_2 + loss_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])

    return losses.avg, loss_names


@torch.no_grad()
def validate(pointnet_cls):
    error_names = ['mAP']
    losses = AverageMeter(i=len(error_names), precision=4)

    pointnet_cls.eval()

    for i, (points, label) in enumerate(val_loader):
        targets, _, _ = pointnet_cls(points)


        losses.update([])

    return losses.avg, error_names


if __name__ == '__main__':
    main()
