_base_ = [
    './upernet_swin_base_patch4_window12_512x512_pretrain_384x384.py',
    '../_base_/datasets/alertwildfire_coco_640x640.py',
    '../_base_/schedules/schedule_160k.py',
]