import argparse
import json
import os
import shutil
import os.path as osp

def main(anno_path, source_dir, target_dir, video_ext, mode):
    data = json.load(open(anno_path, 'r'))

    copy_func = shutil.copy2 if mode == 'copy' else shutil.move

    # Create split folders
    train_dir = osp.join(target_dir, 'training')
    valid_dir = osp.join(target_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Create all the label dirs for both splits
    for label in data['labels']:
        os.makedirs(osp.join(train_dir, label), exist_ok=True)
        os.makedirs(osp.join(valid_dir, label), exist_ok=True)

    # Copy the files according to the annotation file
    for video_id, annotation in data['database'].items():
        source_video_path = osp.join(source_dir, f'{video_id}{video_ext}')
        target_video_dir = osp.join(target_dir,
                                     f'{annotation["subset"]}',
                                     f'{annotation["annotations"]["label"]}',
                                     f'{video_id}{video_ext}')
        copy_func(source_video_path, target_video_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, help='Directory with VIRAT video clips')
    parser.add_argument('--target_dir', type=str, help='Directory where the rearranged clips will be')
    parser.add_argument('--anno_path', type=str, help='Path to json file with video clip annotations')
    parser.add_argument('--mode', type=str, default='copy', help='Determines if file is moved or copied: copy, move')
    parser.add_argument('--video_ext', type=str, default='.mp4', help='Video files extension')
    args = parser.parse_args()

    main(args.anno_path, args.source_dir, args.target_dir, args.video_ext, args.mode)