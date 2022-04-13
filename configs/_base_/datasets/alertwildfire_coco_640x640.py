# dataset settings
dataset_type = 'AlertWildfire'
data_root = 'data/alertwildfire'
coco_root = 'data/coco_stuff164k'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomRotate', prob=0.3, degree=15),
    #dict(type='RandomMosaic', prob=0.1),
    dict(type='RandomCutOut', prob=0.2, n_holes=2, cutout_ratio=[(0.01, 0.01), (0.005, 0.01), (0.01, 0.005), (0.001, 0.001)]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images',
            ann_dir='annotations',
            split='train.lst',
            seg_map_suffix='.png',
            pipeline=train_pipeline,
        ),
        dict(
            type=dataset_type,
            data_root=coco_root,
            img_dir='images/train2017',
            ann_dir='compressed_annotations/train2017',
            split='train.lst',
            seg_map_suffix='_labelTrainIds.png',
            pipeline=train_pipeline,
        )
    ],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        #img_dir='images/training',
        img_dir='images',
        ann_dir='annotations',
        split='val.lst',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        #img_dir='images/validation',
        img_dir='images',
        ann_dir='annotations',
        split='val.lst',
        pipeline=test_pipeline))

