# ======================== 基础配置继承 ========================
_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/schedules/schedule_1x.py', 
    '_base_/default_runtime.py'
]
# 请将这段代码放在配置文件的顶部，例如 _base_ 的下方
custom_imports = dict(
    imports=[
        'denseclip.datasets.mvtec_dataset',  # 导入您的自定义数据集
        'denseclip.models'                   # 导入您提供的 models.py 文件
    ],
    allow_failed_imports=False)

# ======================== 1. 模型修改 ========================
model = dict(
    # 指定CLIP预训练权重的路径 (您的设置是正确的)
    pretrained='pretrained/RN50.pt', 
    backbone=dict(
        type='CLIPResNet',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=800, # 这个值可以根据需要调整，但我们先保持默认
        style='pytorch'),
    roi_head=dict(
        bbox_head=dict(
            # 类别数从COCO的80类改为MVTec的1类
            num_classes=1), 
        mask_head=dict(
            # 类别数从COCO的80类改为MVTec的1类
            num_classes=1)))

# ======================== 2. 数据集修改 ========================
# --- 数据集基本信息 ---
dataset_type = 'CocoDataset'
data_root = 'data/mvtec_coco/'
classes = ('anomaly',) # 明确定义我们的类别元组

# --- 数据处理流水线 (Pipeline) ---
# 这是从coco_instance_clip.py中复制并简化的
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615], std=[68.5005327, 66.6321579, 70.32316305], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# --- 数据加载器 (Dataloader) 配置 ---
data = dict(
    # 根据您的GPU显存调整, 2对于5060来说是安全的起点
    samples_per_gpu=1,
    # 根据您的CPU核心数调整, 2是通用安全值
    workers_per_gpu=2,
    train=dict(
        type='MVTecCocoDataset',
        # 覆盖类别
        classes=classes,
        # 覆盖标注文件路径
        ann_file=data_root + 'annotations/instances_train2017.json',
        # 覆盖图片路径
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])

# ======================== 3. 训练策略修改 ========================
# --- 优化器 ---
# 您的设置是DenseCLIP的推荐设置，非常好，我们保留它
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001,
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                        'text_encoder': dict(lr_mult=0.0), # CLIP的文本编码器不参与训练
                                        'norm': dict(decay_mult=0.)}))

# 修改后
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))

# --- 学习率调度器 ---
# MVTec数据集较小，我们增加训练周期以保证充分收敛
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # 在第16和22个epoch降低学习率
    step=[16, 22])

# --- 运行器 (Runner) ---
# 将总训练周期增加到24个
runner = dict(type='EpochBasedRunner', max_epochs=24)

# ======================== 4. 运行时设置修改 ========================
# 设置checkpoints保存的间隔
checkpoint_config = dict(interval=4) 

# 设置日志打印的间隔
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])