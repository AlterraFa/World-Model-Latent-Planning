import torch
from utils.schedulers import CosineWSDSchedule, CosineWDSchedule, CosineSchedule
from utils.logger import Logger
logger = Logger(__name__)

def compile_opt(apred, lpred, iterations_per_epoch, start_lr, ref_lr, warmup, anneal, num_epochs, wd=1e-06, final_wd=1e-06, final_lr=0.0, mixed_precision=False, betas=(0.9, 0.999), eps=1e-08, zero_init_bias_wd=True, enc_lr_scale=1.0):
    pass
