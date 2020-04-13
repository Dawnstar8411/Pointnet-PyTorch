import argparse

parser=argparse.ArgumentParser(description='Pointnet Classification',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 模型信息
parser.add_argument('--dataset_name', type=str, default="ModelNet40")
parser.add_argument('--model_name', type=str, default="pointnet_cls")
parser.add_argument('--seed', default=2048, type=int, help="seef for random function and network initialization.")

# 读取与保存
parser.add_argument('--data_path', default='/home/yc/chen/data/Pointnet/modelnet40_ply_hdf5_2048/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained', default='./checkpoints/ModelNet40-pointnet_cls/debug/pointnet_cls_256.pth.tar', metavar='PATH')
parser.add_argument('--output_path',default='./outputs')
parser.add_argument('--log_full', default='test_log_full.csv', metavar='PATH')


# 网络测试
parser.add_argument('--no_cuda', default=False, type=bool)

# 具体算法相关
parser.add_argument('--n_pts', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2480]')
parser.add_argument('--num_classes', type =int, default = 40, help = 'number of classes')
# 是否为debug模式
parser.add_argument('--is_debug', type=bool, default=True)

args = parser.parse_args()