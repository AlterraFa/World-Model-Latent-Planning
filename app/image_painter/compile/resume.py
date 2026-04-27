import os
import torch
import glob
from utils.logger import Logger
logger = Logger(__name__)

def load_checkpoint(model, optimizer, checkpoint_dir, checkpoint_name='probe.pt', prefer_best=True, map_location=None):
    pass

def restore_resume_state(resume_meta: dict, scaler, criterion, lr_scheduler, wd_scheduler, start_epoch: int, ipe: int, rank: int, run_idx: int, resume_prefer_best: bool):
    pass

def _load_state_dict_compat(model, state_dict: dict):
    pass

def _normalize_state_dict_for_model(model, state_dict: dict) -> dict:
    pass
