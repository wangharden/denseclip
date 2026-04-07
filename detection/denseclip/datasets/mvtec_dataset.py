import numpy as np
from mmdet.datasets import CocoDataset, DATASETS


@DATASETS.register_module()
class MVTecCocoDataset(CocoDataset):
    """
    专门用于MVTec COCO格式数据集的类。
    这个类的唯一作用就是重写 _filter_imgs 方法，
    以确保在训练时不会因为没有标注而过滤掉任何图片。
    """
    def _filter_imgs(self, min_size=32):
    # 兼容不同版本：data_infos 是较新版本常用的字段，img_infos 也可能存在
        infos = getattr(self, 'data_infos', None) or getattr(self, 'img_infos', None)

    # 如果没有任何 infos，就尝试使用 img_ids（极端兼容）
        if infos is None:
            img_ids = getattr(self, 'img_ids', None)
            if img_ids is None:
                return []  # 没有任何信息，返回空列表（不会被过滤）
        # 返回所有索引
            return list(range(len(img_ids)))

        valid_inds = []
        for i, info in enumerate(infos):
            if min_size > 0:
            # info 可能是 dict，包含 width/height；用 get 防止 KeyError
                width = info.get('width', 0) if isinstance(info, dict) else getattr(info, 'width', 0)
                height = info.get('height', 0) if isinstance(info, dict) else getattr(info, 'height', 0)
                if width > min_size and height > min_size:
                    valid_inds.append(i)
            else:
                valid_inds.append(i)
        return valid_inds
