import csv
import warnings

import h5py
import numpy as np
import torch
from PIL import Image
from path import Path

import models
from config.args_pointnet_cls_test import *
from utils import pc_util
from utils.utils import save_path_formatter

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
args.save_path = args.output_path / save_path
args.save_path.makedirs_p()
print("=> will save everything to {}".format(args.save_path))

print("2.Data Loading...")

shape_names_list = [line.rstrip() for line in open(Path(args.data_path) / 'shape_names.txt', "r")]

test_list = [Path(args.data_path) / line.rstrip() for line in open(Path(args.data_path) / 'test_files.txt', "r")]

test_points = []
test_label = []

for i in np.arange(len(test_list)):
    h5_filename = test_list[i]
    f = h5py.File(h5_filename)
    test_points.append(f['data'][:])
    test_label.append(f['label'][:])

test_points = np.concatenate(test_points, axis=0)  # (num_samples,n_pts,3)
test_label = np.concatenate(test_label, axis=0)  # (num_samples,1)


print("3.Creating Model")

pointnet_cls = models.PointNet_cls(num_cls=args.num_cls, input_transform=True, feature_transform=True).to(device)

if args.pretrained:
    print('=> using pre-trained weights for PoseNet')
    weights = torch.load(args.pretrained)
    pointnet_cls.load_state_dict(weights['state_dict'], strict=False)

print("4. Create csvfile to save log information")

with open(args.save_path / args.log_full, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['PointNet Classification Accuracy Evaluation!'])

print("5. Start Testing!")


@torch.no_grad()
def main():
    pointnet_cls.eval()
    total_correct = 0
    total_num = 0
    total_num_class = [0 for _ in range(args.num_cls)]
    total_correct_class = [0 for _ in range(args.num_cls)]
    error_cnt = 0
    for index in range(len(test_points)):
        origin_points = test_points[index][0:args.n_pts, :]
        points = np.transpose(origin_points, (1, 0))
        points = np.expand_dims(points, 0)
        label = test_label[index]
        points = torch.from_numpy(points)
        label = torch.from_numpy(label)
        points, label = points.to(device).float(), label.to(device).long()

        targets, _ = pointnet_cls(points)

        pred_val = torch.argmax(targets, 1)

        correct = torch.sum(pred_val == label)

        total_correct += correct.item()
        total_num += 1

        label = label.item()
        pred_val = pred_val.item()

        total_num_class[label] += 1
        total_correct_class[label] += np.sum(pred_val == label)

        if pred_val != label and args.visu:
            img_filename = args.save_path / '{}_label_{}_pred_{}.jpg'.format(error_cnt, shape_names_list[label],
                                                                             shape_names_list[pred_val])
            output_img = pc_util.point_cloud_three_views(origin_points)  # np.squeeze(origin_points)
            im = Image.fromarray(output_img)
            im.save(img_filename)
            error_cnt += 1

    avg_precision = total_correct / total_num
    print('Accuracy:{}'.format(avg_precision))
    with open(args.save_path / args.log_full, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerow(['Accuracy:{}'.format(avg_precision)])

    class_accuracies = np.array(total_correct_class) / np.array(total_num_class, dtype=np.float)
    for i, name in enumerate(shape_names_list):
        print('{}: {}'.format(name, class_accuracies[i]))
        with open(args.save_path / args.log_full, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow(['{}:{}'.format(name, class_accuracies[i])])


if __name__ == '__main__':
    main()
