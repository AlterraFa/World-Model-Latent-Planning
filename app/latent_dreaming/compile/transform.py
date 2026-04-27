from typing import Optional
from utils.logger import Logger
logger = Logger(__name__)

def compile_transform(random_horizontal_flip=True, random_resize_aspect_ratio=(3 / 4, 4 / 3), random_resize_scale=(0.3, 1.0), reprob=0.0, auto_augment=False, motion_shift=False, crop_size=224, normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), pad_frame_count: Optional[int]=None, pad_frame_method: str='circulant'):
    pass
