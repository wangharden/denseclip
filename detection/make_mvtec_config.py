# detection/make_mvtec_config.py
from mmcv import Config
from pathlib import Path

# 模板文件（仓库内现有的）
template = 'detection/configs/retinanet_denseclip_r50_fpn_1x_coco.py'
out_path = 'detection/configs/retinanet_denseclip_r50_fpn_mvtec.py'

# ========== 请修改下面路径和类名为你自己的实际值 ==========
MVTEC_BASE = Path(r'C:/Nottingham/DenseCLIP-master/data/mvtec')  # 推荐用正斜杠或 Path
MVTEC_TRAIN_ANN = str(MVTEC_BASE / 'annotations' / 'instances_train.json')
MVTEC_VAL_ANN = str(MVTEC_BASE / 'annotations' / 'instances_val.json')
MVTEC_TEST_ANN = str(MVTEC_BASE / 'annotations' / 'instances_test.json')

MVTEC_TRAIN_IMG = str(MVTEC_BASE / 'train')
MVTEC_VAL_IMG = str(MVTEC_BASE / 'val')
MVTEC_TEST_IMG = str(MVTEC_BASE / 'test')

# 类别示例 -> 请用你 annotations 中 categories 的真实类别替换
CLASSES = ('class1', 'class2', 'class3')  # <- 替换为实际类别元组
# ========================================================

cfg = Config.fromfile(template)

# 修改 data 字段（兼容多种模板写法）
if hasattr(cfg.data, 'train'):
    cfg.data.train.ann_file = MVTEC_TRAIN_ANN
    cfg.data.train.img_prefix = MVTEC_TRAIN_IMG
    cfg.data.train.classes = CLASSES
else:
    cfg.data['train'] = dict(
        ann_file=MVTEC_TRAIN_ANN,
        img_prefix=MVTEC_TRAIN_IMG,
        classes=CLASSES
    )

if hasattr(cfg.data, 'val'):
    cfg.data.val.ann_file = MVTEC_VAL_ANN
    cfg.data.val.img_prefix = MVTEC_VAL_IMG
    cfg.data.val.classes = CLASSES
else:
    cfg.data['val'] = dict(
        ann_file=MVTEC_VAL_ANN,
        img_prefix=MVTEC_VAL_IMG,
        classes=CLASSES
    )

# test 部分处理（有的模板叫 test，有的叫 test_dataloader）
if hasattr(cfg.data, 'test'):
    cfg.data.test.ann_file = MVTEC_TEST_ANN
    cfg.data.test.img_prefix = MVTEC_TEST_IMG
    cfg.data.test.classes = CLASSES
else:
    cfg.data['test'] = dict(
        ann_file=MVTEC_TEST_ANN,
        img_prefix=MVTEC_TEST_IMG,
        classes=CLASSES
    )

# 调整 num_classes（针对常见的 bbox_head 位置）
num_cls = len(CLASSES)
if 'bbox_head' in cfg.model:
    cfg.model['bbox_head']['num_classes'] = num_cls
else:
    # 兜底遍历更新所有可能的 bbox_head 字段
    def update_num_classes(obj):
        if isinstance(obj, dict):
            if 'bbox_head' in obj and isinstance(obj['bbox_head'], dict):
                obj['bbox_head']['num_classes'] = num_cls
            for v in obj.values():
                update_num_classes(v)
    update_num_classes(cfg.model)

# 调小 samples_per_gpu 以适配单卡小显存
try:
    cfg.data.samples_per_gpu = 2
    cfg.data.workers_per_gpu = 2
except Exception:
    pass

# 工作目录（可自定义）
cfg.work_dir = 'work_dirs/retinanet_denseclip_r50_fpn_mvtec'

cfg.dump(out_path)
print('Wrote config to', out_path)
