import argparse
import os
import shutil
import os.path as osp

def get_class_list(class_path):
    with open(class_path, 'r') as class_file:
        return [line.strip().split()[1] for line in class_file.readlines()]


def main(args):
    copy_func = shutil.copy2 if args.mode == 'copy' else shutil.move

    # Create split folders
    train_dir = osp.join(args.target_dir, 'training')
    valid_dir = osp.join(args.target_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Create all the label dirs for both splits
    for label in get_class_list(args.class_anno_path):
        os.makedirs(osp.join(train_dir, label), exist_ok=True)
        os.makedirs(osp.join(valid_dir, label), exist_ok=True)

    # Copy the files for training files
    with open(args.train_anno_path) as train_file:
        for line in train_file.readlines():
            video_rel_path, label_id = line.strip().split()
            source_video_path = osp.join(args.source_dir, video_rel_path)
            target_video_path = osp.join(args.target_dir, 'training', video_rel_path)
            copy_func(source_video_path, target_video_path)

    # Copy the files for the testing files
    with open(args.train_anno_path) as train_file:
        for line in train_file.readlines():
            video_rel_path = line.strip()
            source_video_path = osp.join(args.source_dir, video_rel_path)
            target_video_path = osp.join(args.target_dir, 'validation', video_rel_path)
            copy_func(source_video_path, target_video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, help='Directory with VIRAT video clips')
    parser.add_argument('--target_dir', type=str, help='Directory where the rearranged clips will be')
    parser.add_argument('--class_anno_path', type=str, help='Path to txt file with all the classes')
    parser.add_argument('--train_anno_path', type=str, help='Path to txt file with training annotations')
    parser.add_argument('--test_anno_path', type=str, help='Path to txt file with testing annotations')
    parser.add_argument('--mode', type=str, default='copy', help='Determines if file is moved or copied: copy, move')
    parser.add_argument('--video_ext', type=str, default='.mp4', help='Video files extension')
    args = parser.parse_args()

    main(args)
