import argparse

parser = argparse.ArgumentParser(description="PointNet for 3D Point Cloud Classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 模型信息
parser.add_argument('--dataset_name', type=str, default="ModelNet40")
parser.add_argument('--model_name', type=str, default="pointnet_cls")
parser.add_argument('--seed', default=2048, type=int, help="seed for random function and network initialization.")

# 读取与保存
parser.add_argument('--data_path', default='/home/yc/chen/data/Point_cloud/modelnet40/hdf5/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained', default=None, metavar='PATH',help="path to pre-trained model.")
parser.add_argument('--log_summary', default='progress_log_summary.csv', metavar='PATH')
parser.add_argument('--log_full', default='progress_log_full.csv', metavar='PATH')


# 网络训练
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N', help="number of data loading workers")
parser.add_argument('--epochs', default=300, type=int, metavar='N', help="number of total epochs to run")
parser.add_argument('--batch_size', default=32, type=int, help="Batch Size during training")
parser.add_argument('--epoch_size', default=1000, type=int, metavar='N', help="manual epoch size")
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help="initial learning rate")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help="momentum for sgd, alpha for adam")
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight_decay', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--decay_step', default=50, type=int, help="Decay step for lr decay")
parser.add_argument('--decay_rate', default=0.7, type=float, help="Decay rate for lr decay")

# 模型超参（网络结构与损失函数）
parser.add_argument('--alpha',default = 0.001, type =float, help="balance parameter of loss function")

# 具体算法相关
parser.add_argument('--n_pts', default=2048,type=int,  help='Point Number')
parser.add_argument('--num_cls',default =40, type = int, help='class number')
# 是否为debug模式
parser.add_argument('--is_debug', type=bool, default=True)

args = parser.parse_args()

