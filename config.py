import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', dest='num_classes', type=int, required=True, help='Number of classes in the dataset')
    parser.add_argument('--dataset_path', dest='dataset', type=str, required=True, help='Path to the dataset, follows UCF-101 folder structure')

    parser.add_argument('--epoch_num', dest='epoch_num', default=40, type=int, help='Max number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int, help='Size of the mini-batch')
    parser.add_argument('--step', dest='step', default=10, type=int, help='Video frame sampling steps')
    parser.add_argument('--num_workers', dest='num_workers', default=8, type=int, help='Number of workers for processing data')
    parser.add_argument('--lr', dest='learning_rate', default=1e-2, type=float, help='Learning rate for training')
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float, help='Momentum used while training')
    parser.add_argument('--wd', dest='weight_decay', default=1e-4, type=float, help='Weight decay while training')
    parser.add_argument('--display', dest='display', default=10, type=int, help='Frequency with which training is displayed')
    parser.add_argument('--val_freq', dest='val_freq', default=5, type=int, help='Validation frequency in epochs')
    parser.add_argument('--finetune', dest='finetune', default=False, type=bool, help='Finetune mode enabled')
    parser.add_argument('--pretrained', dest='pretrained', default=None, type=str, help='Path to pretrained model')
    parser.add_argument('--gpu_id', dest='gpu', default=0, type=str, help='List of gpu ids to be used')
    parser.add_argument('--log_path', dest='log', default='log', type=str, help='Path where the logs will be saved')
    parser.add_argument('--save_path', dest='save_path', default='output', type=str, help='Path where the checkpoints will be saved')
    parser.add_argument('--clip_len', dest='clip_len', default=64, type=int, help='Number of frames to be sampled per clip')
    parser.add_argument('--fsr', dest='frame_sample_rate', default=1, type=int, help='Frame sample rate to generate clips')
    parser.add_argument('--patience', dest='patience', default=10, type=int, help='Patience for early stopping')

    parser.add_argument('--num_exp', dest='num_experiments', default=None, type=int, help='Number of experiments')
    parser.add_argument('--exp_log', dest='exp_log', default=None, type=str, help='Path to save experiments logs')


    args = parser.parse_args()
    args.gpu = [int(gpu_id) for gpu_id in args.gpu.split(',')]
    return args
