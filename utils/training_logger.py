import os
import csv
import sys
from typing import Dict, Optional, List, Literal, Union, Tuple, Any
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from progress_table import ProgressTable
    PROGRESS_TABLE_AVAILABLE = True
except ImportError:
    PROGRESS_TABLE_AVAILABLE = False

IS_KAGGLE_COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'

if IS_KAGGLE_COMMIT:
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm
    
from omegaconf import OmegaConf, DictConfig, ListConfig

class NoOpLogger:
    """
    No-operation logger for distributed training - provides the same interface
    as TrainingLogger but performs no actual logging operations.

    This is used for non-zero ranks in distributed training to avoid:
    - Race conditions in directory creation
    - Duplicate logging output
    - Unnecessary computational overhead

    All methods are no-ops and return appropriate default values.
    """

    def __init__(self, *args, **kwargs):
        self.current_epoch = 0
        self.use_validation = kwargs.get('use_validation', True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def start_training(self, description: str = "Training"):
        pass

    def start_epoch(self, epoch: int, num_batches: int, desc: str = "Training"):
        self.current_epoch = epoch

    def start_phase(self, num_batches: int, desc: str = "Phase"):
        pass

    def batch_iterator(self, dataloader):
        return dataloader

    def log_batch(self, metrics, phase="train", step=None, phase_agnostic=None):
        pass

    def log_epoch(self, train_metrics=None, val_metrics=None, extra_metrics=None):
        pass

    def log_model_graph(self, model, input_sample):
        pass

    def log_histogram(self, tag, values, step=None):
        pass

    def log_table(self, key, columns, rows, step=None):
        pass

    def log_media(self, tag, media, media_type="image", step=None, caption=None):
        pass

    def log_plot(self, key, plot, step=None):
        pass

    def alert(self, title, text, level="WARN"):
        pass

    def get_epoch_metrics(self, phase="val"):
        return {}

    def get_metric(self, metric_name, phase="val"):
        return None

    def save_checkpoint(self, models, optimizer, scheduler=None, filename=None,
                        extra_state=None, log_as_artifact=False):
        pass

    def close(self):
        pass

    def set_validation_mode(self, use_validation: bool):
        self.use_validation = use_validation

    def watch(self, model, log="gradients", log_freq=100):
        pass


class TrainingLogger:
    """
    A training logger integrating TensorBoard and (optionally) Weights & Biases,
    with tqdm / progress-table progress bars.

    Args:
        log_dir: Directory to save TensorBoard logs and checkpoints.
        epochs: Total number of training epochs.
        run_name: Optional name appended to log_dir; also used as the W&B run name.
        metrics_to_track: List of metric names to track (auto-detected if omitted).
        progress_type: "tqdm" or "table".
        use_validation: Set False for self-supervised training (no val metrics).
        save_csv: Master switch for CSV export.
        save_batch_csv: Write per-batch CSV rows.
        save_epoch_csv: Write per-epoch CSV rows.
        log_batch_tensorboard: Log every iteration to TensorBoard (not just epoch averages).
        resume_epoch: Resume training from this epoch (reloads existing CSV rows).

        # W&B-specific
        use_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name (required when use_wandb=True).
        wandb_entity: W&B entity (team / user). Uses default if omitted.
        wandb_config: Dict of hyperparameters to log to wandb.config.
        wandb_tags: List of string tags for the W&B run.
        wandb_notes: Free-text notes attached to the run.
        wandb_mode: "online" | "offline" | "disabled". Useful for air-gapped envs.
        wandb_watch_model: Auto-log gradients / parameters via wandb.watch().
        wandb_watch_log: "gradients" | "parameters" | "all" (passed to wandb.watch).
        wandb_watch_log_freq: How often (in batches) to log watched values.
        wandb_log_model_artifact: Upload checkpoints as W&B Artifacts automatically.
    """

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_value(value: float) -> str:
        if isinstance(value, int):
            return str(value)
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4f}"

    @staticmethod
    def _normalize_csv_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return str(value.detach().cpu().tolist())
        if isinstance(value, (list, tuple, dict)):
            return str(value)
        return value

    @staticmethod
    def _upsert_fieldnames(fieldnames, row):
        fields = list(fieldnames)
        for key in row.keys():
            if key not in fields:
                fields.append(key)
        return fields

    @staticmethod
    def _write_csv(path, fieldnames, rows):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        log_dir: str,
        epochs: int,
        run_name: Optional[str] = None,
        metrics_to_track: Optional[List[str]] = None,
        progress_type: Literal["tqdm", "table"] = "tqdm",
        use_validation: bool = True,
        save_csv: bool = True,
        save_batch_csv: bool = False,
        save_epoch_csv: bool = True,
        log_batch_tensorboard: bool = False,
        resume_epoch: int = 0,
        # W&B
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_notes: Optional[str] = None,
        wandb_mode: str = "online",
        wandb_watch_model: bool = False,
        wandb_watch_log: str = "gradients",
        wandb_watch_log_freq: int = 100,
        wandb_log_model_artifact: bool = False,
    ):
        self.epochs = epochs
        self.current_epoch = resume_epoch
        self.metrics_to_track = metrics_to_track or []
        self.progress_type = progress_type
        self.use_validation = use_validation
        self.save_csv = save_csv
        self.save_batch_csv = save_batch_csv
        self.save_epoch_csv = save_epoch_csv
        self.log_batch_tensorboard = log_batch_tensorboard
        self._global_step = 0

        # W&B config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self._wandb_watch_model = wandb_watch_model
        self._wandb_watch_log = wandb_watch_log
        self._wandb_watch_log_freq = wandb_watch_log_freq
        self._wandb_log_model_artifact = wandb_log_model_artifact

        if use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb not installed, W&B logging disabled. pip install wandb")

        if progress_type == "table" and not PROGRESS_TABLE_AVAILABLE:
            print("Warning: progress-table not installed, falling back to tqdm. pip install progress-table")
            self.progress_type = "tqdm"

        # Setup log directory
        self.log_dir = os.path.join(log_dir, run_name) if run_name else log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "weights"), exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Weights & Biases
        self._wandb_run = None
        if self.use_wandb:
            # 2. Convert wandb_config to a plain dict to avoid the Serialization error
            processed_config = wandb_config
            if isinstance(processed_config, (DictConfig, ListConfig)):
                processed_config = OmegaConf.to_container(processed_config, resolve=True)
            elif isinstance(processed_config, dict):
                # If it's a dict that MIGHT contain OmegaConf sub-items, clean those too
                processed_config = {
                    k: (OmegaConf.to_container(v, resolve=True) if isinstance(v, (DictConfig, ListConfig)) else v)
                    for k, v in processed_config.items()
                }

            resume_mode = "allow" if resume_epoch > 0 else None
            self._wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                config=processed_config or {},  # Use the cleaned config here
                tags=wandb_tags,
                notes=wandb_notes,
                mode=wandb_mode,
                resume=resume_mode,
                dir=self.log_dir,
            )
            # Define custom x-axis so W&B uses our epoch counter everywhere
            wandb.define_metric("epoch")
            wandb.define_metric("Train/*", step_metric="epoch")
            wandb.define_metric("Val/*",   step_metric="epoch")
            wandb.define_metric("Misc/*",  step_metric="epoch")
            wandb.define_metric("Iter_*",  step_metric="global_step")

        # Progress bars
        self.epoch_pbar: Optional[tqdm] = None
        self.batch_pbar: Optional[tqdm] = None
        self.progress_table: Optional["ProgressTable"] = None
        self._table_columns_added = False
        self._table_pbar = None

        # Metric accumulators
        self._train_metrics_accum: Dict[str, list] = defaultdict(list)
        self._val_metrics_accum:   Dict[str, list] = defaultdict(list)
        self._misc_metrics_accum:  Dict[str, list] = defaultdict(list)

        # CSV
        self._csv_batch_rows: List[Dict] = []
        self._csv_epoch_rows: List[Dict] = []
        self._csv_batch_fields: List[str] = []
        self._csv_epoch_fields: List[str] = []
        self._csv_batch_path = os.path.join(self.log_dir, "batch_metrics.csv")
        self._csv_epoch_path = os.path.join(self.log_dir, "epoch_metrics.csv")

        # Resume support
        if resume_epoch > 0:
            self._global_step = resume_epoch  # rough default; overridden below if CSV exists
            if self.save_csv and self.save_epoch_csv and os.path.exists(self._csv_epoch_path):
                with open(self._csv_epoch_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    self._csv_epoch_fields = list(reader.fieldnames or [])
                    for row in reader:
                        try:
                            if int(row.get("epoch", 0)) <= resume_epoch:
                                self._csv_epoch_rows.append(dict(row))
                        except (ValueError, TypeError):
                            self._csv_epoch_rows.append(dict(row))
            if self.save_csv and self.save_batch_csv and os.path.exists(self._csv_batch_path):
                with open(self._csv_batch_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    self._csv_batch_fields = list(reader.fieldnames or [])
                    for row in reader:
                        try:
                            if int(row.get("epoch", 0)) <= resume_epoch:
                                self._csv_batch_rows.append(dict(row))
                        except (ValueError, TypeError):
                            self._csv_batch_rows.append(dict(row))
                self._global_step = len(self._csv_batch_rows)

        # State for table display
        self._current_phase = "train"
        self._num_batches = 0
        self._current_batch = 0

    # ------------------------------------------------------------------ #
    # W&B model watching                                                   #
    # ------------------------------------------------------------------ #

    def watch(
        self,
        model: torch.nn.Module,
        log: str = "gradients",
        log_freq: int = 100,
    ):
        """
        Enable W&B gradient / parameter auto-logging for a model.
        Must be called after wandb.init (i.e. after TrainingLogger.__init__).

        Args:
            model: The model to watch.
            log: "gradients" | "parameters" | "all"
            log_freq: How often (in batches) to log.
        """
        if self.use_wandb and self._wandb_run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)

    # ------------------------------------------------------------------ #
    # Training / epoch / phase lifecycle                                   #
    # ------------------------------------------------------------------ #

    def start_training(self, description: str = "Training"):
        if self.progress_type == "tqdm":
            self.epoch_pbar = tqdm(
                total=self.epochs,
                desc=description,
                position=0,
                leave=True,
                file=sys.stdout,
                mininterval=30,
                initial=self.current_epoch,
            )
        else:
            self.progress_table = ProgressTable(
                num_decimal_places=4,
                pbar_embedded=False,
                pbar_show_progress=True,
                pbar_show_eta=True,
                pbar_show_throughput=True,
            )
            self.progress_table.add_column("Epoch", color="bold")
            self._table_columns_added = False

    def start_epoch(self, epoch: int, num_batches: int, desc: str = "Training"):
        self.current_epoch = epoch
        self._train_metrics_accum.clear()
        self._val_metrics_accum.clear()
        self._misc_metrics_accum.clear()
        self._current_phase = desc.lower()
        self._num_batches = num_batches
        self._current_batch = 0

        if self.progress_type == "tqdm":
            self.batch_pbar = tqdm(
                total=num_batches,
                desc=f"Epoch {self.current_epoch + 1}/{self.epochs} - {desc}",
                position=1,
                leave=False,
                file=sys.stdout,
                mininterval=30,
            )
        else:
            self._table_num_batches = num_batches
            if self.progress_table is not None:
                self.progress_table["Epoch"] = f"{self.current_epoch + 1}/{self.epochs}"

    def start_phase(self, num_batches: int, desc: str = "Phase"):
        self._current_phase = desc.lower()
        self._num_batches = num_batches
        self._current_batch = 0

        if self.progress_type == "tqdm":
            if self.batch_pbar is not None:
                self.batch_pbar.close()
            self.batch_pbar = tqdm(
                total=num_batches,
                desc=f"Epoch {self.current_epoch + 1}/{self.epochs} - {desc}",
                position=1,
                leave=False,
                file=sys.stdout,
                mininterval=30,
            )
        else:
            self._table_num_batches = num_batches

    def batch_iterator(self, dataloader):
        if self.progress_type == "tqdm":
            yield from dataloader
        else:
            if self.progress_table is not None:
                yield from self.progress_table(dataloader)
            else:
                yield from dataloader

    # ------------------------------------------------------------------ #
    # Batch logging                                                        #
    # ------------------------------------------------------------------ #

    def log_batch(
        self,
        metrics: Dict[str, float],
        phase: str = "train",
        step: Optional[int] = None,
        phase_agnostic: Optional[List[str]] = None,
    ):
        """
        Log metrics for a single batch and update progress display.

        Args:
            metrics: Dict of metric names → values.
            phase: "train" or "val".
            step: Optional global step override.
            phase_agnostic: Metric names that don't belong to train/val
                            (e.g. learning rate logged mid-batch).
        """
        phase_agnostic = set(phase_agnostic or [])

        clean_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            clean_metrics[key] = value

        train_or_val_accum = (
            self._train_metrics_accum if phase == "train" else self._val_metrics_accum
        )
        for key, value in clean_metrics.items():
            if key in phase_agnostic:
                self._misc_metrics_accum[key].append(value)
            else:
                train_or_val_accum[key].append(value)

        if self.save_csv and self.save_batch_csv:
            row = {"epoch": self.current_epoch + 1, "batch": self._current_batch + 1, "phase": phase}
            row.update({k: self._normalize_csv_value(v) for k, v in clean_metrics.items()})
            self._csv_batch_rows.append(row)
            self._csv_batch_fields = self._upsert_fieldnames(self._csv_batch_fields, row)
            self._write_csv(self._csv_batch_path, self._csv_batch_fields, self._csv_batch_rows)

        self._current_batch += 1
        self._global_step += 1

        if self.log_batch_tensorboard:
            tb_prefix = "Iter_Train" if phase == "train" else "Iter_Val"
            for key, value in clean_metrics.items():
                if key in phase_agnostic:
                    self.writer.add_scalar(f"Iter_Misc/{key}", value, self._global_step)
                else:
                    self.writer.add_scalar(f"{tb_prefix}/{key}", value, self._global_step)

        # W&B per-iteration logging
        if self.use_wandb and self._wandb_run is not None and self.log_batch_tensorboard:
            wb_prefix = "Iter_Train" if phase == "train" else "Iter_Val"
            wb_log: Dict[str, Any] = {"global_step": self._global_step}
            for key, value in clean_metrics.items():
                bucket = "Iter_Misc" if key in phase_agnostic else wb_prefix
                wb_log[f"{bucket}/{key}"] = value
            wandb.log(wb_log, step=self._global_step)

        # Progress display
        if self.progress_type == "tqdm":
            if self.batch_pbar is not None:
                display_metrics = {k: self._format_value(v) for k, v in metrics.items()}
                self.batch_pbar.set_postfix(display_metrics)
                self.batch_pbar.update(1)
        else:
            if self.progress_table is not None:
                if not self._table_columns_added:
                    for key in clean_metrics.keys():
                        if key in phase_agnostic:
                            self.progress_table.add_column(f"{key}", color="magenta")
                        elif self.use_validation:
                            self.progress_table.add_column(f"Train | {key}", color="blue")
                            self.progress_table.add_column(f"Val | {key}", color="green")
                        else:
                            self.progress_table.add_column(key, color="blue")
                    self._table_columns_added = True

                for key in clean_metrics.keys():
                    if key in phase_agnostic:
                        avg = sum(self._misc_metrics_accum[key]) / len(self._misc_metrics_accum[key])
                        self.progress_table[f"{key}"] = self._format_value(avg)
                    else:
                        prefix = ("Train | " if phase == "train" else "Val | ") if self.use_validation else ""
                        avg = sum(train_or_val_accum[key]) / len(train_or_val_accum[key])
                        self.progress_table[f"{prefix}{key}" if prefix else key] = self._format_value(avg)

    # ------------------------------------------------------------------ #
    # Epoch logging                                                        #
    # ------------------------------------------------------------------ #

    def log_epoch(
        self,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log epoch-level metrics to TensorBoard (and W&B) and advance progress.

        Args:
            train_metrics: Override for training metrics (uses accumulated batch avg if None).
            val_metrics: Override for validation metrics (uses accumulated batch avg if None).
            extra_metrics: Additional metrics, e.g. learning rate, EMA momentum.
        """
        epoch = self.current_epoch + 1  # 1-indexed for display

        if train_metrics is None:
            train_metrics = {k: sum(v) / len(v) for k, v in self._train_metrics_accum.items() if v}
        if val_metrics is None and self.use_validation:
            val_metrics = {k: sum(v) / len(v) for k, v in self._val_metrics_accum.items() if v}
        elif val_metrics is None:
            val_metrics = {}

        misc_metrics = {k: sum(v) / len(v) for k, v in self._misc_metrics_accum.items() if v}

        # --- TensorBoard ---
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"Train/{key}", _to_scalar(value), epoch)
        if self.use_validation and val_metrics:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"Val/{key}", _to_scalar(value), epoch)
        if misc_metrics:
            for key, value in misc_metrics.items():
                self.writer.add_scalar(f"Misc/{key}", _to_scalar(value), epoch)
        if extra_metrics:
            for key, value in extra_metrics.items():
                self.writer.add_scalar(f"Misc/{key}", _to_scalar(value), epoch)
        self.writer.flush()

        # --- W&B ---
        if self.use_wandb and self._wandb_run is not None:
            wb_log: Dict[str, Any] = {"epoch": epoch}
            for key, value in train_metrics.items():
                wb_log[f"Train/{key}"] = _to_scalar(value)
            if self.use_validation and val_metrics:
                for key, value in val_metrics.items():
                    wb_log[f"Val/{key}"] = _to_scalar(value)
            for key, value in misc_metrics.items():
                wb_log[f"Misc/{key}"] = _to_scalar(value)
            if extra_metrics:
                for key, value in extra_metrics.items():
                    wb_log[f"Misc/{key}"] = _to_scalar(value)
            wandb.log(wb_log, step=epoch)

            # Update run summary with latest metrics (useful for W&B table comparisons)
            for key, value in train_metrics.items():
                wandb.run.summary[f"Train/{key}"] = _to_scalar(value)
            if self.use_validation and val_metrics:
                for key, value in val_metrics.items():
                    wandb.run.summary[f"Val/{key}"] = _to_scalar(value)

        # --- CSV ---
        if self.save_csv and self.save_epoch_csv:
            row: Dict[str, Any] = {"epoch": epoch}
            for key, value in train_metrics.items():
                row[f"Train/{key}"] = self._normalize_csv_value(value)
            if self.use_validation and val_metrics:
                for key, value in val_metrics.items():
                    row[f"Val/{key}"] = self._normalize_csv_value(value)
            for key, value in misc_metrics.items():
                row[f"Misc/{key}"] = self._normalize_csv_value(value)
            if extra_metrics:
                for key, value in extra_metrics.items():
                    row[f"Misc/{key}"] = self._normalize_csv_value(value)
            self._csv_epoch_rows.append(row)
            self._csv_epoch_fields = self._upsert_fieldnames(self._csv_epoch_fields, row)
            self._write_csv(self._csv_epoch_path, self._csv_epoch_fields, self._csv_epoch_rows)

        # --- Progress display ---
        if self.progress_type == "tqdm":
            if self.batch_pbar is not None:
                self.batch_pbar.close()
                self.batch_pbar = None
            if self.epoch_pbar is not None:
                self.epoch_pbar.update(1)
        else:
            if self.progress_table is not None:
                if extra_metrics and not hasattr(self, '_extra_columns_added'):
                    for key in extra_metrics.keys():
                        color = "yellow" if "lr" in key.lower() else "magenta"
                        self.progress_table.add_column(f"{key}", color=color)
                    self._extra_columns_added = True
                if extra_metrics:
                    for key, value in extra_metrics.items():
                        self.progress_table[f"{key}"] = self._format_value(
                            value.item() if isinstance(value, torch.Tensor) else value
                        )
                self.progress_table.next_row()

        self._print_epoch_summary(epoch, train_metrics, val_metrics, extra_metrics)

    # ------------------------------------------------------------------ #
    # Rich W&B logging                                                     #
    # ------------------------------------------------------------------ #

    def log_table(
        self,
        key: str,
        columns: List[str],
        rows: List[List[Any]],
        step: Optional[int] = None,
    ):
        """
        Log a structured table to W&B (e.g. predictions vs ground truth).

        Example:
            logger.log_table(
                "predictions",
                columns=["image", "pred", "label", "confidence"],
                rows=[[wandb.Image(img), pred, label, conf], ...],
            )

        Args:
            key: W&B table key shown in the dashboard.
            columns: Column headers.
            rows: List of rows (each row is a list matching columns).
            step: Optional step override (uses current epoch if omitted).
        """
        if not (self.use_wandb and self._wandb_run is not None):
            return
        table = wandb.Table(columns=columns, data=rows)
        wandb.log({key: table}, step=step or (self.current_epoch + 1))

    def log_media(
        self,
        tag: str,
        media: Any,
        media_type: Literal["image", "audio", "video", "html", "molecule"] = "image",
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ):
        """
        Log rich media to TensorBoard and/or W&B.

        For W&B, wraps the value in the appropriate wandb media class.
        For TensorBoard, only images are supported (others are W&B-only).

        Args:
            tag: Metric key / panel name.
            media: A torch.Tensor (image), file path (str), or pre-built wandb object.
            media_type: One of "image", "audio", "video", "html", "molecule".
            step: Optional step (defaults to current epoch).
            caption: Optional caption (W&B only).
        """
        step = step or (self.current_epoch + 1)

        # TensorBoard: images only
        if media_type == "image" and isinstance(media, torch.Tensor):
            self.writer.add_image(tag, media, step)

        # W&B
        if self.use_wandb and self._wandb_run is not None:
            if not isinstance(media, (wandb.Image, wandb.Audio, wandb.Video,
                                      wandb.Html, wandb.Molecule)):
                kwargs = {"caption": caption} if caption else {}
                if media_type == "image":
                    media = wandb.Image(media, **kwargs)
                elif media_type == "audio":
                    media = wandb.Audio(media, **kwargs)
                elif media_type == "video":
                    media = wandb.Video(media, **kwargs)
                elif media_type == "html":
                    media = wandb.Html(media)
                elif media_type == "molecule":
                    media = wandb.Molecule(media, **kwargs)
            wandb.log({tag: media}, step=step)

    def log_plot(
        self,
        key: str,
        plot: Any,
        step: Optional[int] = None,
    ):
        """
        Log a pre-built wandb.plot.* chart.

        Example:
            from sklearn.metrics import confusion_matrix
            import wandb
            cm = wandb.plot.confusion_matrix(
                y_true=labels, preds=preds, class_names=["cat", "dog"]
            )
            logger.log_plot("confusion_matrix", cm)

        Args:
            key: Chart name in the W&B dashboard.
            plot: A wandb.plot object.
            step: Optional step override.
        """
        if not (self.use_wandb and self._wandb_run is not None):
            return
        wandb.log({key: plot}, step=step or (self.current_epoch + 1))

    def alert(
        self,
        title: str,
        text: str,
        level: Literal["INFO", "WARN", "ERROR"] = "WARN",
    ):
        """
        Send a W&B alert (email / Slack) when a notable event occurs.

        Example:
            if val_loss < best_loss:
                logger.alert("New best model", f"Val loss dropped to {val_loss:.4f}")

        Args:
            title: Short title for the alert.
            text: Alert body text.
            level: "INFO", "WARN", or "ERROR".
        """
        if self.use_wandb and self._wandb_run is not None:
            wandb.alert(title=title, text=text, level=level)

    # ------------------------------------------------------------------ #
    # Existing helpers                                                     #
    # ------------------------------------------------------------------ #

    def log_model_graph(
        self,
        model: torch.nn.Module,
        input_sample: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    ):
        """Log model architecture graph to TensorBoard and W&B."""
        # 1. TensorBoard Graph (Tracing)
        if self.use_wandb and self._wandb_run is not None:
            self.watch(model, log="all")

            # 3. Log a textual summary as a W&B Table (Very readable)
            try:
                model_stats = []
                for name, module in model.named_modules():
                    # Only log top-level or main layers to keep table clean
                    if len(name.split('.')) <= 2: 
                        params = sum(p.numel() for p in module.parameters())
                        model_stats.append([name, str(type(module).__name__), params])
                
                self.log_table(
                    "model_structure", 
                    columns=["Layer Name", "Layer Type", "Parameters"], 
                    rows=model_stats
                )
                
                # Also save the raw string representation
                with open(os.path.join(self.log_dir, "model_summary.txt"), "w") as f:
                    f.write(str(model))
                    
            except Exception as e:
                print(f"W&B Table logging failed: {e}")

        # 2. Weights & Biases Architecture Logging
        if self.use_wandb and self._wandb_run is not None:
            try:
                # Use your existing watch method to track gradients/parameters
                # This populates the 'Gradients' and 'Graph' tabs in W&B
                self.watch(model, log=self._wandb_watch_log, log_freq=self._wandb_watch_log_freq)

                # Log the text representation of the model as an Artifact
                # This is useful for auditing the exact architecture layers later
                arch_path = os.path.join(self.log_dir, "model_arch.txt")
                with open(arch_path, "w") as f:
                    f.write(str(model))
                
                model_artifact = wandb.Artifact(
                    name=f"{self._wandb_run.name or 'model'}_architecture", 
                    type="model_description"
                )
                model_artifact.add_file(arch_path)
                self._wandb_run.log_artifact(model_artifact)

            except Exception as e:
                msg = f"Warning: Could not log W&B model graph: {e}"
                tqdm.write(msg) if self.progress_type == "tqdm" else print(msg)

    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Log a histogram of values (e.g. weights, gradients) to TensorBoard."""
        step = step or self.current_epoch + 1
        self.writer.add_histogram(tag, values, step)
        if self.use_wandb and self._wandb_run is not None:
            wandb.log({tag: wandb.Histogram(values.detach().cpu().numpy())}, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None):
        """Log an image to TensorBoard (and W&B if enabled)."""
        self.log_media(tag, image, media_type="image", step=step)

    def get_epoch_metrics(self, phase: str = "val") -> Dict[str, float]:
        accum = (self._train_metrics_accum if phase == "train"
                 else self._misc_metrics_accum if phase == "misc"
                 else self._val_metrics_accum)
        return {k: sum(v) / len(v) for k, v in accum.items() if v}

    def get_metric(self, metric_name: str, phase: str = "val") -> Optional[float]:
        accum = (self._train_metrics_accum if phase == "train"
                 else self._misc_metrics_accum if phase == "misc"
                 else self._val_metrics_accum)
        if metric_name in accum and accum[metric_name]:
            return sum(accum[metric_name]) / len(accum[metric_name])
        return None

    def save_checkpoint(
        self,
        models: Dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        filename: Optional[str] = None,
        extra_state: Optional[Dict] = None,
        log_as_artifact: Optional[bool] = None,
    ):
        """
        Save a training checkpoint and (optionally) upload it as a W&B Artifact.

        Args:
            models: Dict of name → model.
            optimizer: The optimizer.
            scheduler: Optional LR scheduler.
            filename: Override default filename.
            extra_state: Additional state to include in the checkpoint dict.
            log_as_artifact: Upload to W&B as a versioned Artifact.
                             Defaults to wandb_log_model_artifact constructor arg.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"

        checkpoint: Dict[str, Any] = {
            "epoch": self.current_epoch,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        for name, model in models.items():
            checkpoint[f"{name}_state_dict"] = model.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if extra_state:
            checkpoint.update(extra_state)

        save_path = os.path.join(self.log_dir, "weights", filename)
        torch.save(checkpoint, save_path)

        msg = f"Checkpoint saved: {save_path}"
        tqdm.write(msg) if self.progress_type == "tqdm" else print(msg)

        # W&B artifact upload
        should_upload = log_as_artifact if log_as_artifact is not None else self._wandb_log_model_artifact
        if should_upload and self.use_wandb and self._wandb_run is not None:
            artifact = wandb.Artifact(
                name=f"checkpoint-epoch-{self.current_epoch + 1}",
                type="model",
                description=f"Checkpoint at epoch {self.current_epoch + 1}",
                metadata={"epoch": self.current_epoch + 1},
            )
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, extra_metrics=None):
        if self.progress_type == "table":
            return
        parts = [f"Epoch {epoch}/{self.epochs}"]
        for key, value in train_metrics.items():
            label = f"Train {key}" if self.use_validation else key
            parts.append(f"{label}: {self._format_value(value)}")
        if self.use_validation and val_metrics:
            for key, value in val_metrics.items():
                parts.append(f"Val {key}: {self._format_value(value)}")
        if extra_metrics:
            for key, value in extra_metrics.items():
                parts.append(f"{key}: {self._format_value(value)}")
        tqdm.write(" | ".join(parts))

    def set_validation_mode(self, use_validation: bool):
        self.use_validation = use_validation

    def close(self):
        if self.progress_type == "tqdm":
            if self.batch_pbar is not None:
                self.batch_pbar.close()
            if self.epoch_pbar is not None:
                self.epoch_pbar.close()
        else:
            if self.progress_table is not None:
                self.progress_table.close()
        self.writer.close()
        if self.use_wandb and self._wandb_run is not None:
            wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception during training: {exc_type.__name__}: {exc_val}")
        self.close()
        return False


# ------------------------------------------------------------------ #
# Module-level helpers                                                 #
# ------------------------------------------------------------------ #

def _to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)


def get_next_run(exp_dir: str) -> int:
    os.makedirs(exp_dir, exist_ok=True)
    existing_runs = [
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d)) and d.startswith("run")
    ]
    if not existing_runs:
        return 1
    run_numbers = []
    for run in existing_runs:
        try:
            run_numbers.append(int(run.replace("run", "")))
        except ValueError:
            continue
    return max(run_numbers) + 1 if run_numbers else 1


def create_supervised_logger(
    log_dir: str,
    epochs: int,
    run_name: Optional[str] = None,
    progress_type: Literal["tqdm", "table"] = "tqdm",
    save_csv: bool = True,
    save_batch_csv: bool = False,
    save_epoch_csv: bool = True,
    log_batch_tensorboard: bool = False,
    resume_epoch: int = 0,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    **wandb_kwargs,
) -> TrainingLogger:
    """Create a TrainingLogger configured for supervised training (with validation)."""
    return TrainingLogger(
        log_dir=log_dir,
        epochs=epochs,
        run_name=run_name,
        progress_type=progress_type,
        use_validation=True,
        save_csv=save_csv,
        save_batch_csv=save_batch_csv,
        save_epoch_csv=save_epoch_csv,
        log_batch_tensorboard=log_batch_tensorboard,
        resume_epoch=resume_epoch,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
        **wandb_kwargs,
    )


def create_self_supervised_logger(
    log_dir: str,
    epochs: int,
    run_name: Optional[str] = None,
    progress_type: Literal["tqdm", "table"] = "tqdm",
    save_csv: bool = True,
    save_batch_csv: bool = False,
    save_epoch_csv: bool = True,
    log_batch_tensorboard: bool = False,
    resume_epoch: int = 0,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    **wandb_kwargs,
) -> TrainingLogger:
    """Create a TrainingLogger configured for self-supervised training (no validation)."""
    return TrainingLogger(
        log_dir=log_dir,
        epochs=epochs,
        run_name=run_name,
        progress_type=progress_type,
        use_validation=False,
        save_csv=save_csv,
        save_batch_csv=save_batch_csv,
        save_epoch_csv=save_epoch_csv,
        log_batch_tensorboard=log_batch_tensorboard,
        resume_epoch=resume_epoch,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
        **wandb_kwargs,
    )


# ------------------------------------------------------------------ #
# Test                                                                 #
# ------------------------------------------------------------------ #

def test_logger(progress_type: str = "tqdm", use_validation: bool = True):
    import tempfile, time
    mode_str = "with validation" if use_validation else "no validation (self-supervised)"
    print("=" * 70)
    print(f"Testing TrainingLogger  progress_type='{progress_type}'  {mode_str}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(
            log_dir=tmpdir,
            epochs=5,
            run_name=f"test_{progress_type}",
            progress_type=progress_type,
            use_validation=use_validation,
            # W&B disabled in tests; set use_wandb=True + wandb_project="my-proj" to enable
            use_wandb=False,
        )

        with logger:
            logger.start_training(f"Testing ({progress_type} mode)")

            for epoch in range(5):
                fake_train = range(20)
                fake_val = range(8)

                logger.start_epoch(epoch, len(fake_train), desc="Training")
                for _ in logger.batch_iterator(fake_train):
                    logger.log_batch(
                        {"Loss": 1.0 - epoch * 0.1, "L1": 0.5 - epoch * 0.05},
                        phase="train",
                    )
                    time.sleep(0.005)

                if use_validation:
                    logger.start_phase(len(fake_val), desc="Validation")
                    for _ in logger.batch_iterator(fake_val):
                        logger.log_batch(
                            {"Loss": 1.1 - epoch * 0.15, "L1": 0.55 - epoch * 0.06},
                            phase="val",
                        )
                        time.sleep(0.005)

                logger.log_epoch(
                    extra_metrics={"LearningRate": 1e-3 * (0.9 ** epoch)}
                )

    print(f"\nTest done — logs: {tmpdir}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if mode in ("tqdm", "table"):
        test_logger(mode, use_validation=True)
        test_logger(mode, use_validation=False)
    else:
        test_logger("tqdm", use_validation=True)
        test_logger("tqdm", use_validation=False)
        test_logger("table", use_validation=True)
        test_logger("table", use_validation=False)