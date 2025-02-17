import os
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
from lib import slowfastnet
from tensorboardX import SummaryWriter
from config import parse_opts

# TODO: remove from being global
params = parse_opts()

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ys = np.zeros(len(train_dataloader.dataset))
    ys_ = np.zeros((len(train_dataloader.dataset), params.num_classes))
    count = 0

    model.train()
    end = time.time()
    print('---Training----------------------------------------------------')
    for step, (inputs, labels) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        bz = inputs.size()[0]
        ys[count:count + bz] = labels.data.numpy().squeeze()

        inputs = inputs.cuda(params.gpu[0])
        labels = labels.cuda(params.gpu[0])
        logits = model(inputs)
        loss = criterion(logits, labels)

        # Requirements for mAP
        _, preds = torch.max(logits, 1)
        probs = torch.softmax(logits.squeeze(), dim=-1)
        ys_[count:count + bz] = probs.data.cpu().numpy().squeeze()
        count += bz

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits.data, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % params.display == 0:
            print(f'Epoch {epoch} [{step + 1}/{len(train_dataloader)}]  loss: {losses.avg:.5f}  Top-1 acc: {top1.avg:.2f}  Top-5 acc: {top5.avg:.2f}')

    ys = ys[:count].astype(int)
    ys_ = ys_[:count]
    ll = np.squeeze(np.eye(params.num_classes)[ys.reshape(-1)])
    ap_acc = 0
    class_ap = []
    for class_idx in range(params.num_classes - 1):
        y_pred = ys_[:, class_idx]
        y_true = ll[:, class_idx]
        ap = average_precision_score(y_true, y_pred)
        class_ap.append(ap)
        ap_acc += ap
    train_map = ap_acc / (params.num_classes - 1)

    print(f'Training: Epoch {epoch} loss: {losses.avg:.5f}  Top-1 acc: {top1.avg:.2f}  Top-5 acc: {top5.avg:.2f}')
    print(f'Class AP:\n{class_ap}')
    print(f'Training mAP: {train_map}    samples: {count}')
    write_exp_log(f'[{epoch}] Training mAP: {train_map}    samples: {count}')

    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)

    return top1.avg, top5.avg, losses.avg

def validation(model, val_dataloader, epoch, criterion, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    ys = np.zeros(len(val_dataloader.dataset))
    ys_ = np.zeros((len(val_dataloader.dataset), params.num_classes))
    count = 0

    end = time.time()
    print('---Validation----------------------------------------------------')
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            data_time.update(time.time() - end)

            bz = inputs.size()[0]
            ys[count:count + bz] = labels.data.numpy().squeeze()

            inputs = inputs.cuda(params.gpu[0])
            labels = labels.cuda(params.gpu[0])
            logits = model(inputs)
            loss = criterion(logits, labels)

            # Requirements for mAP
            _, preds = torch.max(logits, 1)
            probs = torch.softmax(logits.squeeze(), dim=-1)
            ys_[count:count + bz] = probs.data.cpu().numpy().squeeze()
            count += bz

            # measure accuracy and record loss

            prec1, prec5 = accuracy(logits.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (step + 1) % params.display == 0:
                print(f'Epoch {epoch} [{step + 1}/{len(val_dataloader)}]  loss: {losses.avg:.5f}  Top-1 acc: {top1.avg:.2f}  Top-5 acc: {top5.avg:.2f}')

    ys = ys[:count].astype(int)
    ys_ = ys_[:count]
    ll = np.squeeze(np.eye(params.num_classes)[ys.reshape(-1)])
    ap_acc = 0
    class_ap = []
    for class_idx in range(params.num_classes - 1):
        y_pred = ys_[:, class_idx]
        y_true = ll[:, class_idx]
        ap = average_precision_score(y_true, y_pred)
        class_ap.append(ap)
        ap_acc += ap
    train_map = ap_acc / (params.num_classes - 1)

    print(f'Validation: Epoch {epoch}  loss: {losses.avg:.05f}  Top-1 acc: {top1.avg:.02f}  Top-5 acc: {top5.avg:.02f}')
    print(f'Class AP:\n{class_ap}')
    print(f'Validation mAP: {train_map}  samples: {count}')
    write_exp_log(f'[{epoch}] Validation mAP: {train_map}  samples: {count}')

    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)

    return top1.avg, top5.avg, losses.avg


def main():
    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params.log, cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")
    train_dataloader = \
        DataLoader(
            VideoDataset(params.dataset, mode='train', clip_len=params.clip_len, frame_sample_rate=params.frame_sample_rate),
            batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

    val_dataloader = \
        DataLoader(
            VideoDataset(params.dataset, mode='validation', clip_len=params.clip_len, frame_sample_rate=params.frame_sample_rate),
            batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

    print("load model")
    model = slowfastnet.resnet50(class_num=params.num_classes)
    
    if params.pretrained is not None:
        pretrained_dict = torch.load(params.pretrained, map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("Load pretrain model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model = model.cuda(params.gpu[0])
    model = nn.DataParallel(model, device_ids=params.gpu)  # multi-Gpu

    criterion = nn.CrossEntropyLoss().cuda(params.gpu[0])
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.step, gamma=0.1)

    model_save_dir = os.path.join(params.save_path, cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    best_loss = 1e6
    best_valid = dict(top1_acc=0., top5_acc=0., epoch=0)
    no_loss_decrease_count = 0

    for epoch in range(params.epoch_num):
        train_top1_acc, train_top5_acc, train_loss = train(model, train_dataloader, epoch, criterion, optimizer, writer)

        if (epoch+1) % params.val_freq == 0:
            val_top1_acc, val_top5_acc, val_loss = validation(model, val_dataloader, epoch, criterion, writer)

            if val_top1_acc > best_valid['top1_acc']:
                best_valid['top1_acc'] = val_top1_acc
                best_valid['top5_acc'] = val_top5_acc
                best_valid['epoch'] = epoch

        if train_loss < best_loss:
            best_loss = train_loss
            no_loss_decrease_count = 0
        else:
            no_loss_decrease_count += 1
        if no_loss_decrease_count >= params.patience:
            print(f'Early stop on Epoch {epoch} with patience {params.patience}')
            write_exp_log(f'[{epoch}] Early stop')
            break

        scheduler.step()

        if epoch % 1 == 0:
            checkpoint = os.path.join(model_save_dir,
                                      "clip_len_" + str(params.clip_len) + "frame_sample_rate_" +str(params.frame_sample_rate) + "_checkpoint_" + str(epoch) + ".pth.tar")
            torch.save(model.module.state_dict(), checkpoint)

    print(f'Best Validated model was found on epoch {best_valid["epoch"]}:  Top1 acc: {best_valid["top1_acc"]}  Top5 acc: {best_valid["top5_acc"]}')
    write_exp_log(f'Best model found on epoch {best_valid["epoch"]}:  Top1 acc: {best_valid["top1_acc"]}  Top5 acc: {best_valid["top5_acc"]}')

    writer.close()


def get_experiment_idx(trackbook_path):
    return 0 if not os.path.exists(trackbook_path) else int(open(trackbook_path, 'r').readline())


def inc_experiment_idx(trackbook_path):
    next_idx = 1
    if os.path.exists(trackbook_path):
        next_idx = get_experiment_idx(trackbook_path) + 1
    open(trackbook_path, 'w').write(str(next_idx))
    return next_idx


def load_experiment_params(exp_params_path):
    return np.load(exp_params_path)


def print_experiment_params(exp_id, exp_params):
    print(f'Experiment {exp_id}:')
    print(f'valid: {exp_params.val_freq}')
    print(f'lr:    {exp_params.learning_rate:.8f}')
    print(f'm:     {exp_params.momentum:.8f}')
    print(f'wd:    {exp_params.weight_decay:.8f}')


def write_exp_log(msg):
    if params.exp_log is not None:
        with open(params.exp_log, 'a+') as exp_log:
            exp_log.write(f'{msg}\n')


if __name__ == '__main__':
    if params.num_experiments is None:
        main()
    else:
        trackbook_path = 'trackbook.txt'
        exp_params = load_experiment_params('exp_params.npy')
        exp_id = get_experiment_idx(trackbook_path)

        while exp_id < params.num_experiments:
            params.learning_rate, params.momentum, params.weight_decay = exp_params[exp_id]
            print_experiment_params(exp_id, params)
            params.exp_log = f'output/experiments/{exp_id:02d}.log'
            try:
                main()
            except ValueError:
                print(f'Experiment {exp_id} failed!')
                with open('output/experiments/failed.log', 'a+') as fail_log:
                    fail_log.write(f'{exp_id}\n')
            exp_id = inc_experiment_idx(trackbook_path)
