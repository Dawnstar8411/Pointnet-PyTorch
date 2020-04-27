import argparse

parser = argparse.ArgumentParser(description='Pointnet Semantic Segmentation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 模型信息
parser.add_argument('--dataset_name', type=str, default="ShapeNet")
parser.add_argument('--model_name', type=str, default="pointnet_part_seg")
parser.add_argument('--seed', default=2048, type=int, help="seed for random function and network initialization.")

# 读取与保存
parser.add_argument('--data_path', default='/home/yc/chen/data/Point_cloud/modelnet40/hdf5/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained', default='./checkpoints/ModelNet40-pointnet_cls/xxx/xxx', metavar='PATH')
parser.add_argument('--output_path', default='./outputs')
parser.add_argument('--log_full', default='test_log_full.csv', metavar='PATH')

# 网络测试
parser.add_argument('--no_cuda', default=False, type=bool)

# 具体算法相关
parser.add_argument('--n_pts', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2480]')
parser.add_argument('--num_cls', type=int, default=16, help="number of classes")
parser.add_argument('--num_parts', type=int, default=50, help='number of parts')
# 是否为debug模式
parser.add_argument('--is_debug', type=bool, default=True)

args = parser.parse_args()
