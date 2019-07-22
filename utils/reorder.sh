#!/usr/bin/env bash

# This script reorder the VIRAT video clips to follow the codebase structure
# for training the SlowFast model
# author: Salvador Medina

source_dir=/home/zal/Data/VIRAT/clean_clips
target_dir=/home/zal/Data/VIRAT/slowfast_clips
anno_path=/home/zal/Data/VIRAT/sp1.json
video_ext=.mp4
mode=copy #can be move

python3 reorder_virat_clips.py --source_dir=${source_dir} --target_dir=${target_dir} --anno_path=${anno_path} --video_ext=${video_ext} --mode=${mode}