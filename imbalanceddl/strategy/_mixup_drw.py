import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer

from imbalanceddl.utils.utils import AverageMeter
from imbalanceddl.utils.metrics import accuracy


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MixupTrainer(Trainer):
    """Mixup-DRW Trainer
    ...
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NEW: keep history of per-class weights (numpy arrays)
        self.per_cls_weights_history = []

    def get_criterion(self):
        if self.strategy == 'Mixup_DRW':
            if self.cfg.epochs == 300:
                idx = self.epoch // 250
            else:
                idx = self.epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(
                self.cls_num_list)
            per_cls_weights_t = torch.FloatTensor(per_cls_weights).cuda(
                self.cfg.gpu)

            # Log as before
            print("=> Per Class Weight = {}".format(per_cls_weights_t))

            # NEW: record history (CPU numpy for easy aggregation later)
            self.per_cls_weights_history.append(per_cls_weights.astype(np.float64))

            self.criterion = nn.CrossEntropyLoss(weight=per_cls_weights_t,
                                                 reduction='none').cuda(
                                                     self.cfg.gpu)
        else:
            raise ValueError("[Warning] Strategy is not supported !")

    # NEW: utility to aggregate class weights across epochs
    def get_class_weight_distribution(self, agg: str = 'sum', normalize: bool = False):
        """
        Aggregate per-class weights across calls to get_criterion().

        Parameters
        ----------
        agg : {'sum', 'mean'}
            How to aggregate the history across epochs.
        normalize : bool
            If True, normalize the aggregated weights to sum to 1.

        Returns
        -------
        np.ndarray
            Aggregated (and optionally normalized) per-class weights, shape (num_classes,)
        """
        if not self.per_cls_weights_history:
            raise RuntimeError("No per-class weights recorded yet. Call get_criterion() at least once.")

        W = np.vstack(self.per_cls_weights_history)  # (num_records, C)
        if agg == 'sum':
            vec = W.sum(axis=0)
        elif agg == 'mean':
            vec = W.mean(axis=0)
        else:
            raise ValueError("agg must be 'sum' or 'mean'")

        if normalize:
            s = vec.sum()
            if s > 0:
                vec = vec / s
        return vec

    # NEW: save to CSV with multiple helpful columns
    def save_class_weight_distribution(self, path: str, agg: str = 'sum', normalize: bool = False):
        """
        Save per-class weight distribution to CSV.

        Columns:
        - class_index
        - last_epoch_weight
        - agg_weight (sum or mean)
        - normalized_weight (if normalize=True)

        Parameters
        ----------
        path : str
            Output CSV path (e.g., 'class_weight_distribution.csv')
        agg : {'sum', 'mean'}
            Aggregation across epochs.
        normalize : bool
            Whether to include a normalized column.
        """
        import csv
        if not self.per_cls_weights_history:
            raise RuntimeError("No per-class weights recorded yet. Call get_criterion() at least once.")

        last_w = self.per_cls_weights_history[-1]
        agg_w = self.get_class_weight_distribution(agg=agg, normalize=False)

        norm_w = None
        if normalize:
            s = agg_w.sum()
            norm_w = agg_w / s if s > 0 else agg_w

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['class_index', 'last_epoch_weight', f'agg_weight_{agg}']
            if normalize:
                header.append('normalized_weight')
            writer.writerow(header)

            for c in range(len(last_w)):
                row = [c, float(last_w[c]), float(agg_w[c])]
                if normalize:
                    row.append(float(norm_w[c]))
                writer.writerow(row)

        # Optional: also log a quick summary to the training log if available
        summary = (f"=> Saved class weight distribution to {path} | "
                   f"agg={agg}, normalize={normalize}")
        print(summary)
        if hasattr(self, 'log_training') and self.log_training is not None:
            self.log_training.write(summary + '\n')
            self.log_training.flush()

    def train_one_epoch(self):
        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # for confusion matrix
        all_preds = list()
        all_targets = list()

        # switch to train mode
        self.model.train()

        for i, (_input, target) in enumerate(self.train_loader):

            if self.cfg.gpu is not None:
                _input = _input.cuda(self.cfg.gpu, non_blocking=True)
                target = target.cuda(self.cfg.gpu, non_blocking=True)

            # Mixup Data
            _input_mix, target_a, target_b, lam = mixup_data(_input, target)
            # Two kinds of output
            output_prec, _ = self.model(_input)
            output_mix, _ = self.model(_input_mix)

            # For Loss, we use mixup output
            loss = mixup_criterion(self.criterion, output_mix, target_a,
                                   target_b, lam).mean()
            acc1, acc5 = accuracy(output_prec, target, topk=(1, 5))
            _, pred = torch.max(output_prec, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure accuracy and record loss
            losses.update(loss.item(), _input.size(0))
            top1.update(acc1[0], _input.size(0))
            top5.update(acc5[0], _input.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.cfg.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              self.epoch,
                              i,
                              len(self.train_loader),
                              loss=losses,
                              top1=top1,
                              top5=top5,
                              lr=self.optimizer.param_groups[-1]['lr'] * 0.1))
                print(output)
                self.log_training.write(output + '\n')
                self.log_training.flush()

        self.compute_metrics_and_record(all_preds,
                                        all_targets,
                                        losses,
                                        top1,
                                        top5,
                                        flag='Training')
