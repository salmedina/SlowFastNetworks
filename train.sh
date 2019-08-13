#!/bin/bash

python3 -u train.py --num_classes=20 \
--dataset_path=/mnt/ssd2tb/salvadom/Data/VIRAT/slowfast_clips \
--lr=1e-2 \
--momentum=0.9 \
--wd=1e-4 \
--gpu_id=0,1,2,3 \
--log_path=log \
--save_path=VIRAT
