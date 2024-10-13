import numpy as np
import argparse



def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pred_path', type=str, default='')
    parser.add_argument('--gt_path', type=str, default='/inputs/Projects/PerDet/datasets/PerSeg/Annotations')

    parser.add_argument('--ref_idx', type=str, default='00')

    args = parser.parse_args()
    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)

    area_intersection = np.logical_and(output, target).sum()
    area_union = np.logical_or(output, target).sum()
    area_target = target.sum()

    return area_intersection, area_union, area_target