import numpy as np


class runningScoreSeg(object):
    # Adapted from https://github.com/meetshah1995/pytorch-semseg
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):  # label_true / label_pred: length (width*height)
        mask = (label_true >= 0) & (label_true < n_class)  # remove invalid class (class 250)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)   # [n_classes, n_classes]
        return hist

    def update(self, gt, pred):  # [batch_size, height, width]
        for lt, lp in zip(gt, pred):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwaviu = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW IoU : \t": fwaviu,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScoreDepth(object):
    def __init__(self, dataset):
        self.error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        self.metric_len = len(self.error_names)
        self.error_metric = [0 for i in range(self.metric_len)]  # [0,0,0,0,...,0]
        self.dataset = dataset
        self.reset()

    def compute_errors_depth(self, gt, pred, crop=True):  # input gt, pred: numpy array, shape: [batch_size, h, w]
        abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
        batch_size = gt.shape[0]

        '''
        crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        construct a mask of False values, with the same size as target
        and then set to True values inside the crop
        '''
        if crop:
            crop_mask = gt[0] != gt[0]
            if self.dataset == 'kitti':
                y1, y2 = int(0.40810811 * gt.shape[1]), int(0.99189189 * gt.shape[1])
                x1, x2 = int(0.03594771 * gt.shape[2]), int(0.96405229 * gt.shape[2])
            elif self.dataset == 'cityscapes':
                y1, y2 = int(0.05 * gt.shape[1]), int(0.80 * gt.shape[1])
                x1, x2 = int(0.05 * gt.shape[2]), int(0.99 * gt.shape[2])
            crop_mask[y1:y2, x1:x2] = 1

        for current_gt, current_pred in zip(gt, pred):  # for each image in a batch
            valid = (current_gt > 0) & (current_gt < 80) & (current_pred > 0) & (
                        current_pred < 80)  # mask out depth not in (0, 80)
            if crop:
                valid = valid & crop_mask

            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid]

            # valid_pred = valid_pred * np.median(valid_gt)/np.median(valid_pred)

            thresh = np.maximum((valid_gt / valid_pred), (valid_pred / valid_gt))
            a1 += (thresh < 1.25).mean()
            a2 += (thresh < 1.25 ** 2).mean()
            a3 += (thresh < 1.25 ** 3).mean()

            rmse += np.sqrt(np.mean((valid_gt - valid_pred) ** 2))
            rmse_log += np.sqrt(np.mean((np.log(valid_gt) - np.log(valid_pred)) ** 2))

            abs_diff += np.mean(np.abs(valid_gt - valid_pred))
            abs_rel += np.mean(np.abs(valid_gt - valid_pred) / valid_gt)

            sq_rel += np.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

        return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]]

    def update(self, gt, pred):
        self.error_metric = self.compute_errors_depth(gt, pred)
        n = 1
        self.count += n
        for i, v in enumerate(self.error_metric):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def reset(self):
        self.val = [0.0]*self.metric_len
        self.avg = [0.0]*self.metric_len
        self.sum = [0.0]*self.metric_len
        self.count = 0

    def get_scores(self):
        return ({
                "abs diff: \t": self.avg[0],
                "abs rel : \t": self.avg[1],
                "sq rel : \t": self.avg[2],
                "rmse : \t": self.avg[3],
                "rmse log : \t": self.avg[4],
                "threshold 1 : \t": self.avg[5],
                "threshold 2 : \t": self.avg[6],
                "threshold 3 : \t": self.avg[7]}
        )


class averageMeter(object):
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

