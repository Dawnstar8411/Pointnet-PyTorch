import datetime

from path import Path


def save_path_formatter(args):
    args_dict = vars(args)
    dataset_name = str(Path(args_dict['dataset_name']))
    folder_string = [dataset_name]
    folder_string.append(str(Path(args_dict['model_name'])))
    save_path = Path('-'.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    if args.is_debug:
        return save_path / 'debug'
    else:
        return save_path / timestamp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert (len(val) == self.meters)
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)