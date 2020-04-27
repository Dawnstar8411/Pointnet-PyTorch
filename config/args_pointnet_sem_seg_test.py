import argparse

parser=argparse.ArgumentParser(description='Pointnet Semantic Segmentation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 模型信息
parser.add_argument('--dataset_name', type=str, default="Indoor3d")
parser.add_argument('--model_name', type=str, default="pointnet_sem_seg")
parser.add_argument('--seed', default=2048, type=int, help="seed for random function and network initialization.")

# 读取与保存
parser.add_argument('--data_path', default='/home/yc/chen/data/Poin_cloud/indoor3d/hdf5/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained', default='./checkpoints/', metavar='PATH')
parser.add_argument('--output_path',default='./outputs')
parser.add_argument('--log_full', default='test_log_full.csv', metavar='PATH')


# 网络测试
parser.add_argument('--no_cuda', default=False, type=bool)

# 具体算法相关
parser.add_argument('--n_pts', type=int, default=4096, help='Point Number')
parser.add_argument('--num_classes', type =int, default = 13, help = 'number of classes')
parser.add_argument('--output_filelist',default = None, help='')
parser.add_argument('--room_data_filelist',default = None, help='')
parser.add_argument('--no_clutter',default= True, help='')

# 是否为debug模式
parser.add_argument('--is_debug', type=bool, default=True)

args = parser.parse_args()