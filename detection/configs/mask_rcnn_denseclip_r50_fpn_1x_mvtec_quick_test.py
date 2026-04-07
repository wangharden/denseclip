# ======================== 快速测试配置 (1小时内出结果) ========================
_base_ = './mask_rcnn_denseclip_r50_fpn_1x_mvtec.py'

# 只训练 1 个 epoch 用于快速验证
runner = dict(type='EpochBasedRunner', max_epochs=1)

# 使用更大的 batch size 加快速度 (如果 GPU 内存允许)
data = dict(
    samples_per_gpu=4,  # 增加到 4 (原来可能是 2)
    workers_per_gpu=2,
)

# 减少验证频率
evaluation = dict(interval=1, metric=['bbox', 'segm'])

# 更快的学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,  # 减少预热步数
    warmup_ratio=0.001,
    step=[1])  # 只在第1个epoch结束时降低学习率

# 减少日志输出频率
log_config = dict(
    interval=20,  # 每20个iteration输出一次
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 每500个iteration保存一次checkpoint
checkpoint_config = dict(interval=1)  # 每个epoch结束保存

print("=" * 60)
print("  快速测试模式")
print("  - 只训练 1 个 epoch")
print("  - 预计时间: 10-30 分钟 (GPU) 或 3-7 小时 (CPU)")
print("=" * 60)
