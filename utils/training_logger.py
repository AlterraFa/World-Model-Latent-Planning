import os
import csv
import sys
import socket
from abc import ABC
from typing import Dict, Optional, List, Literal, Union, Tuple, Any
from collections import defaultdict

import torch

from .logger import Logger
logger = Logger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from progress_table import ProgressTable
    PROGRESS_TABLE_AVAILABLE = True
except ImportError:
    PROGRESS_TABLE_AVAILABLE = False

IS_KAGGLE = bool(os.environ.get('KAGGLE_URL_BASE') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
IS_KAGGLE_COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'

if IS_KAGGLE_COMMIT:
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm

from omegaconf import OmegaConf, DictConfig, ListConfig


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def _to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)


def _has_internet(host: str = "api.wandb.ai", port: int = 443, timeout: float = 2.0) -> bool:
    """Return True when a short TCP probe succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ================================================================== #
# Logging Backends                                                     #
# ================================================================== #

class LoggerBackend(ABC):
    """
    Abstract base for a logging backend.

    All methods are no-ops by default so concrete backends only override
    what they support.  TrainingLogger dispatches every call to each
    registered backend in order.
    """

    def log_epoch_scalars(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log epoch-level scalars.  Keys use ``Train/``, ``Val/`` etc. prefixes."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, epoch)

    def log_iter_scalars(self, metrics: Dict[str, float], global_step: int) -> None:
        """Log per-iteration scalars.  Keys use ``Iter_Train/`` etc. prefixes."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, global_step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    def log_histogram(self, tag: str, values, step: int) -> None:
        pass

    def log_media(self, tag: str, media, media_type: str, step: int,
                  caption: Optional[str] = None) -> None:
        pass

    def log_table(self, key: str, columns: List[str], rows: List[List[Any]],
                  step: int) -> None:
        pass

    def log_plot(self, key: str, plot: Any, step: int) -> None:
        pass

    def alert(self, title: str, text: str, level: str = "WARN") -> None:
        pass

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        pass

    def log_artifact(self, path: str, name: str, artifact_type: str = "model",
                     metadata: Optional[Dict] = None) -> None:
        pass

    def save_checkpoint_artifact(self, path: str, name: str,
                                  metadata: Optional[Dict] = None) -> None:
        """Upload checkpoint according to the backend's own config.  No-op by default."""
        pass

    def update_summary(self, metrics: Dict[str, float]) -> None:
        """Update run-level summary (e.g. best metrics).  No-op by default."""
        pass

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ #
# TensorBoard backend                                                  #
# ------------------------------------------------------------------ #

class TensorBoardBackend(LoggerBackend):
    """Writes scalars, histograms, and images to a TensorBoard ``SummaryWriter``."""

    def __init__(self, log_dir: str):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("tensorboard is not installed.  pip install tensorboard")
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_epoch_scalars(self, metrics: Dict[str, float], epoch: int) -> None:
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, epoch)
        self._writer.flush()

    def log_iter_scalars(self, metrics: Dict[str, float], global_step: int) -> None:
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, global_step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        self._writer.add_histogram(tag, values, step)

    def log_media(self, tag: str, media, media_type: str, step: int,
                  caption: Optional[str] = None) -> None:
        if media_type == "image" and isinstance(media, torch.Tensor):
            self._writer.add_image(tag, media, step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


# ------------------------------------------------------------------ #
# Weights & Biases backend                                             #
# ------------------------------------------------------------------ #

class WandbBackend(LoggerBackend):
    """
    Writes metrics, media, tables, and artifacts to Weights & Biases.

    Args:
        project: W&B project name.
        run_name: Display name for the run.
        entity: W&B team / user (uses default if omitted).
        config: Hyperparameters dict logged to ``wandb.config``.
        tags: List of string tags.
        notes: Free-text notes for the run.
        mode: ``"online"`` | ``"offline"`` | ``"disabled"``.
        log_dir: Local directory for W&B files.
        resume: ``"allow"`` | ``"must"`` | ``"never"`` | ``None``.
        log_model_artifact: Upload checkpoints as W&B Artifacts automatically.
        watch_log: ``"gradients"`` | ``"parameters"`` | ``"all"`` for ``wandb.watch``.
        watch_log_freq: How often (in batches) ``wandb.watch`` logs values.
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        log_dir: Optional[str] = None,
        resume: Optional[str] = None,
        log_model_artifact: bool = False,
        watch_log: str = "gradients",
        watch_log_freq: int = 100,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed.  pip install wandb")

        # Auto-switch to offline when running in non-networked environments,
        # Kaggle Batch kernels, or when explicitly requested.
        offline_env = (
            IS_KAGGLE
            or os.environ.get("WANDB_MODE", "").lower() == "offline"
            or not _has_internet()
        )
        if mode == "online" and offline_env:
            logger.INFO("WandbBackend: no internet/offline env detected - switching to mode='offline'")
            mode = "offline"

        # Flatten OmegaConf objects so wandb can serialise them
        if isinstance(config, (DictConfig, ListConfig)):
            config = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, dict):
            config = {
                k: (OmegaConf.to_container(v, resolve=True)
                    if isinstance(v, (DictConfig, ListConfig)) else v)
                for k, v in config.items()
            }

        init_kwargs = dict(
            project=project,
            entity=entity,
            name=run_name,
            config=config or {},
            tags=tags,
            notes=notes,
            mode=mode,
            resume=resume,
            dir=log_dir,
            save_code=False,
        )
        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as e:
            if mode == "online":
                logger.WARNING(
                    "WandbBackend: online init failed, retrying offline. "
                    f"Reason: {type(e).__name__}: {e}"
                )
                init_kwargs["mode"] = "offline"
                self._run = wandb.init(**init_kwargs)
            else:
                raise
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        wandb.define_metric("Train/*", step_metric="epoch")
        wandb.define_metric("Val/*",   step_metric="epoch")
        wandb.define_metric("Misc/*",  step_metric="epoch")
        wandb.define_metric("Iter_*",  step_metric="global_step")

        self.log_model_artifact = log_model_artifact
        self._watch_log = watch_log
        self._watch_log_freq = watch_log_freq

    def log_epoch_scalars(self, metrics: Dict[str, float], epoch: int) -> None:
        wandb.log({"epoch": epoch, **metrics})
        # Keep run summary up to date for cross-run comparisons
        summary = {k: v for k, v in metrics.items()
                   if k.startswith(("Train/", "Val/"))}
        if summary:
            self.update_summary(summary)

    def log_iter_scalars(self, metrics: Dict[str, float], global_step: int) -> None:
        wandb.log({"global_step": global_step, **metrics})

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        wandb.log({"global_step": step, tag: value})

    def log_histogram(self, tag: str, values, step: int) -> None:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        wandb.log({"global_step": step, tag: wandb.Histogram(values)})

    def log_media(self, tag: str, media, media_type: str, step: int,
                  caption: Optional[str] = None) -> None:
        if isinstance(media, (wandb.Image, wandb.Audio, wandb.Video,
                               wandb.Html, wandb.Molecule)):
            wandb.log({"global_step": step, tag: media})
            return
        kw = {"caption": caption} if caption else {}
        _wrap = {
            "image":    lambda m: wandb.Image(m, **kw),
            "audio":    lambda m: wandb.Audio(m, **kw),
            "video":    lambda m: wandb.Video(m, **kw),
            "html":     lambda m: wandb.Html(m),
            "molecule": lambda m: wandb.Molecule(m, **kw),
        }
        wandb.log({"global_step": step, tag: _wrap.get(media_type, lambda m: m)(media)})

    def log_table(self, key: str, columns: List[str], rows: List[List[Any]],
                  step: int) -> None:
        wandb.log({"global_step": step, key: wandb.Table(columns=columns, data=rows)})

    def log_plot(self, key: str, plot: Any, step: int) -> None:
        wandb.log({"global_step": step, key: plot})

    def alert(self, title: str, text: str, level: str = "WARN") -> None:
        wandb.alert(title=title, text=text, level=level)

    def watch_model(self, model, log: str = None, log_freq: int = None) -> None:
        wandb.watch(model,
                    log=log or self._watch_log,
                    log_freq=log_freq or self._watch_log_freq)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model",
                     metadata: Optional[Dict] = None) -> None:
        artifact = wandb.Artifact(name=name, type=artifact_type,
                                  metadata=metadata or {})
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def save_checkpoint_artifact(self, path: str, name: str,
                                  metadata: Optional[Dict] = None) -> None:
        if self.log_model_artifact:
            self.log_artifact(path, name, artifact_type="model", metadata=metadata)

    def update_summary(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self._run.summary[key] = value

    def close(self) -> None:
        wandb.finish()


# ------------------------------------------------------------------ #
# MLflow backend                                                       #
# ------------------------------------------------------------------ #

class MLflowBackend(LoggerBackend):
    """
    Writes metrics, params, tags, and artifacts to MLflow.

    Args:
        experiment_name: MLflow experiment to log under.  Created if it
            does not exist yet.
        run_name: Display name for the MLflow run.
        tracking_uri: MLflow tracking server URI.  Defaults to the
            ``MLFLOW_TRACKING_URI`` environment variable or a local
            ``./mlruns`` directory.
        tags: Extra key-value tags attached to the run.
        params: Hyperparameters logged once at startup via
            ``mlflow.log_params``.
        log_model_artifact: Upload checkpoints as MLflow artifacts.
        run_id: Resume an existing run by ID instead of creating a new one.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        log_model_artifact: bool = False,
        run_id: Optional[str] = None,
        offline: bool = False,
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow is not installed.  pip install mlflow")

        # Offline mode: force a local file-based tracking URI so no server is needed.
        if offline or IS_KAGGLE_COMMIT:
            if tracking_uri is None:
                tracking_uri = "./mlruns"
            logger.INFO(f"MLflowBackend: offline mode — tracking URI set to '{tracking_uri}'")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        self._run = mlflow.start_run(run_id=run_id, run_name=run_name, tags=tags)
        self.log_model_artifact = log_model_artifact

        if params:
            # mlflow.log_params has a 100-param-per-call limit; chunk if needed
            items = list(params.items())
            for i in range(0, len(items), 100):
                mlflow.log_params(dict(items[i : i + 100]))

    # -- properties --------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    # -- logging -----------------------------------------------------------

    def log_epoch_scalars(self, metrics: Dict[str, float], epoch: int) -> None:
        mlflow.log_metrics(metrics, step=epoch)

    def log_iter_scalars(self, metrics: Dict[str, float], global_step: int) -> None:
        mlflow.log_metrics(metrics, step=global_step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        mlflow.log_metric(tag, value, step=step)

    def log_media(self, tag: str, media, media_type: str, step: int,
                  caption: Optional[str] = None) -> None:
        """Save tensors/paths as image artifacts.  Only images are supported."""
        if media_type != "image":
            return
        try:
            import tempfile
            from torchvision.utils import save_image
            if isinstance(media, torch.Tensor):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmp_path = f.name
                save_image(media.float(), tmp_path)
                mlflow.log_artifact(tmp_path, artifact_path=f"media/{tag}")
                os.remove(tmp_path)
        except Exception as e:
            logger.WARNING(f"MLflowBackend.log_media failed: {e}")

    def log_artifact(self, path: str, name: str, artifact_type: str = "model",
                     metadata: Optional[Dict] = None) -> None:
        mlflow.log_artifact(path, artifact_path=artifact_type)

    def save_checkpoint_artifact(self, path: str, name: str,
                                  metadata: Optional[Dict] = None) -> None:
        if self.log_model_artifact:
            self.log_artifact(path, name, artifact_type="checkpoints")

    def update_summary(self, metrics: Dict[str, float]) -> None:
        # MLflow has no explicit run-summary; log with step=-1 as a convention
        mlflow.log_metrics({f"summary/{k}": v for k, v in metrics.items()}, step=-1)

    def close(self) -> None:
        mlflow.end_run()


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
    Training logger with pluggable backends (TensorBoard, W&B, …) and
    tqdm / progress-table progress bars.

    Args:
        log_dir: Directory to save CSV files and checkpoints.  If
            ``run_name`` is also given the final path is
            ``os.path.join(log_dir, run_name)``.
        epochs: Total number of training epochs.
        backends: List of :class:`LoggerBackend` instances to dispatch
            every logging call to.  Defaults to
            ``[TensorBoardBackend(log_dir)]`` when omitted.
        run_name: Optional sub-directory name appended to ``log_dir``.
        metrics_to_track: Metric names to track (auto-detected if omitted).
        progress_type: ``"tqdm"`` or ``"table"``.
        use_validation: Set ``False`` for self-supervised training.
        save_csv: Master switch for CSV export.
        save_batch_csv: Write per-batch CSV rows.
        save_epoch_csv: Write per-epoch CSV rows.
        log_batch_tensorboard: Forward every iteration to backends (not
            just epoch averages).
        resume_epoch: Resume training from this epoch (reloads CSV rows).
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
        backends: Optional[List[LoggerBackend]] = None,
        run_name: Optional[str] = None,
        metrics_to_track: Optional[List[str]] = None,
        progress_type: Literal["tqdm", "table"] = "tqdm",
        use_validation: bool = True,
        save_csv: bool = True,
        save_batch_csv: bool = False,
        save_epoch_csv: bool = True,
        log_batch_scalars: bool = False,
        resume_epoch: int = 0,
    ):
        self.epochs = epochs
        self.current_epoch = resume_epoch
        self.metrics_to_track = metrics_to_track or []
        self.progress_type = progress_type
        self.use_validation = use_validation
        self.save_csv = save_csv
        self.save_batch_csv = save_batch_csv
        self.save_epoch_csv = save_epoch_csv
        self.log_batch_tensorboard = log_batch_scalars
        self._global_step = 0

        if progress_type == "table" and not PROGRESS_TABLE_AVAILABLE:
            logger.WARNING("progress-table not installed, falling back to tqdm. pip install progress-table")
            self.progress_type = "tqdm"

        # Setup log directory
        self.log_dir = os.path.join(log_dir, run_name) if run_name else log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "weights"), exist_ok=True)

        # Backends — default to TensorBoard only
        if backends is None:
            backends = [TensorBoardBackend(self.log_dir)]
        self._backends: List[LoggerBackend] = list(backends)

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
    # Model watching                                                       #
    # ------------------------------------------------------------------ #

    def watch(
        self,
        model: torch.nn.Module,
        log: str = "gradients",
        log_freq: int = 100,
    ):
        """
        Forward model watching to all backends (e.g. W&B gradient logging).

        Args:
            model: The model to watch.
            log: ``"gradients"`` | ``"parameters"`` | ``"all"``
            log_freq: How often (in batches) to log.
        """
        for backend in self._backends:
            backend.watch_model(model, log=log, log_freq=log_freq)

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
            iter_scalars: Dict[str, float] = {}
            for key, value in clean_metrics.items():
                bucket = "Iter_Misc" if key in phase_agnostic else tb_prefix
                iter_scalars[f"{bucket}/{key}"] = value
            for backend in self._backends:
                backend.log_iter_scalars(iter_scalars, self._global_step)

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

        # --- Build prefixed scalars dict and dispatch to all backends ---
        epoch_scalars: Dict[str, float] = {}
        for key, value in train_metrics.items():
            epoch_scalars[f"Train/{key}"] = _to_scalar(value)
        if self.use_validation and val_metrics:
            for key, value in val_metrics.items():
                epoch_scalars[f"Val/{key}"] = _to_scalar(value)
        for key, value in misc_metrics.items():
            epoch_scalars[f"Misc/{key}"] = _to_scalar(value)
        if extra_metrics:
            for key, value in extra_metrics.items():
                epoch_scalars[f"Misc/{key}"] = _to_scalar(value)

        for backend in self._backends:
            backend.log_epoch_scalars(epoch_scalars, epoch)

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
    # Rich logging                                                         #
    # ------------------------------------------------------------------ #

    def log_table(
        self,
        key: str,
        columns: List[str],
        rows: List[List[Any]],
        step: Optional[int] = None,
    ):
        """
        Log a structured table to all backends that support it (e.g. W&B).

        Example::

            logger.log_table(
                "predictions",
                columns=["image", "pred", "label", "confidence"],
                rows=[[wandb.Image(img), pred, label, conf], ...],
            )

        Args:
            key: Table name shown in the dashboard.
            columns: Column headers.
            rows: List of rows (each row is a list matching columns).
            step: Optional step override (uses current epoch if omitted).
        """
        step = step or (self.current_epoch + 1)
        for backend in self._backends:
            backend.log_table(key, columns, rows, step)

    def log_media(
        self,
        tag: str,
        media: Any,
        media_type: Literal["image", "audio", "video", "html", "molecule"] = "image",
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ):
        """
        Log rich media to all backends.

        Args:
            tag: Metric key / panel name.
            media: A ``torch.Tensor`` (image), file path (str), or a pre-built
                backend media object.
            media_type: One of ``"image"``, ``"audio"``, ``"video"``,
                ``"html"``, ``"molecule"``.
            step: Optional step (defaults to current epoch).
            caption: Optional caption (backend-specific).
        """
        step = step or (self.current_epoch + 1)
        for backend in self._backends:
            backend.log_media(tag, media, media_type, step, caption)

    def log_plot(
        self,
        key: str,
        plot: Any,
        step: Optional[int] = None,
    ):
        """
        Log a pre-built chart object to all backends that support it.

        Example (W&B)::

            cm = wandb.plot.confusion_matrix(
                y_true=labels, preds=preds, class_names=["cat", "dog"]
            )
            logger.log_plot("confusion_matrix", cm)

        Args:
            key: Chart name in the backend dashboard.
            plot: A backend-specific plot object.
            step: Optional step override.
        """
        step = step or (self.current_epoch + 1)
        for backend in self._backends:
            backend.log_plot(key, plot, step)

    def alert(
        self,
        title: str,
        text: str,
        level: Literal["INFO", "WARN", "ERROR"] = "WARN",
    ):
        """
        Send an alert through all backends that support it (e.g. W&B email/Slack).

        Args:
            title: Short title for the alert.
            text: Alert body text.
            level: ``"INFO"``, ``"WARN"``, or ``"ERROR"``.
        """
        for backend in self._backends:
            backend.alert(title, text, level)

    # ------------------------------------------------------------------ #
    # Existing helpers                                                     #
    # ------------------------------------------------------------------ #

    def log_model_graph(
        self,
        model: torch.nn.Module,
        input_sample: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]] = None,
    ):
        """Save model architecture to disk and notify all backends."""
        arch_path = os.path.join(self.log_dir, "model_arch.txt")
        try:
            with open(arch_path, "w") as f:
                f.write(str(model))
        except Exception as e:
            logger.WARNING(f"Could not save model arch text: {e}")

        # Log a layer-summary table for backends that support it (e.g. W&B)
        try:
            model_stats = [
                [name, type(module).__name__,
                 sum(p.numel() for p in module.parameters())]
                for name, module in model.named_modules()
                if len(name.split(".")) <= 2
            ]
            self.log_table(
                "model_structure",
                columns=["Layer Name", "Layer Type", "Parameters"],
                rows=model_stats,
            )
        except Exception as e:
            logger.WARNING(f"Could not log model structure table: {e}")

        for backend in self._backends:
            try:
                backend.watch_model(model)
            except Exception as e:
                logger.WARNING(f"{type(backend).__name__}.watch_model failed: {e}")
            try:
                backend.log_artifact(arch_path, name="model_architecture",
                                     artifact_type="model_description")
            except Exception as e:
                logger.WARNING(f"{type(backend).__name__}.log_artifact failed: {e}")

    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Log a histogram of values (e.g. weights, gradients) to all backends."""
        step = step or self.current_epoch + 1
        for backend in self._backends:
            backend.log_histogram(tag, values, step)

    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None):
        """Log an image to all backends."""
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
        Save a training checkpoint and optionally upload it via all backends.

        Args:
            models: Dict of name → model.
            optimizer: The optimizer.
            scheduler: Optional LR scheduler.
            filename: Override default filename.
            extra_state: Additional state to include in the checkpoint dict.
            log_as_artifact: ``True`` forces upload; ``False`` skips upload;
                ``None`` lets each backend decide (via
                :meth:`LoggerBackend.save_checkpoint_artifact`).
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
        tqdm.write(msg) if self.progress_type == "tqdm" else logger.INFO(msg)

        ckpt_name = f"checkpoint-epoch-{self.current_epoch + 1}"
        ckpt_meta = {"epoch": self.current_epoch + 1}
        if log_as_artifact is True:
            for backend in self._backends:
                backend.log_artifact(save_path, ckpt_name,
                                     artifact_type="model", metadata=ckpt_meta)
        elif log_as_artifact is None:
            for backend in self._backends:
                backend.save_checkpoint_artifact(save_path, ckpt_name,
                                                  metadata=ckpt_meta)

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
        for backend in self._backends:
            backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.ERROR(f"Exception during training: {exc_type.__name__}: {exc_val}")
        self.close()
        return False


# ------------------------------------------------------------------ #
# Module-level helpers                                                 #
# ------------------------------------------------------------------ #


def build_backends(
    backends_cfg: List[Dict[str, Any]],
    runtime_params: Optional[Dict[str, Any]] = None,
    force_offline: Optional[bool] = None,
) -> List[LoggerBackend]:
    """
    Instantiate a list of :class:`LoggerBackend` objects from config dicts.

    Each entry in *backends_cfg* must follow the ``autoload_modules``
    convention::

        target: utils.training_logger.TensorBoardBackend
        params:
          log_dir: ./runs/run1

    ``runtime_params`` are merged into every backend's ``params`` before
    instantiation, but *only* for keys whose config value is ``null``.
    Any other YAML value (string, int, bool, …) is kept as-is.
    This lets callers inject dynamic values (``run_dir``, ``run_name``, …)
    while still allowing per-backend overrides in YAML.

    Example YAML block (inside ``logging:``):::

        backends:
          - target: utils.training_logger.TensorBoardBackend
            params:
              log_dir: null             # filled from runtime_params
          - target: utils.training_logger.WandbBackend
            params:
              project: my_project       # kept as-is from YAML
              run_name: null            # filled from runtime_params
              log_dir: null             # filled from runtime_params
          - target: utils.training_logger.MLflowBackend
            params:
              experiment_name: my_experiment
              run_name: null            # filled from runtime_params

    Args:
        backends_cfg: List of ``{target, params}`` dicts (from OmegaConf or plain dict).
        runtime_params: Key/value pairs injected into each backend's params when
            the key is already present in the config (placeholder value is
            overwritten regardless of what it was).
    Returns:
        List of instantiated :class:`LoggerBackend` objects.
    """
    from utils.autoload_modules import get_obj_from_str

    runtime_params = runtime_params or {}
    # If not explicitly set, auto-detect offline environments (Kaggle batch, etc.)
    if force_offline is None:
        force_offline = IS_KAGGLE_COMMIT or os.environ.get("WANDB_MODE", "").lower() == "offline"
    backends: List[LoggerBackend] = []

    for cfg in backends_cfg:
        if isinstance(cfg, (DictConfig, ListConfig)):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        target = cfg.get("target")
        if not target:
            raise KeyError(f"Backend config entry is missing 'target': {cfg}")

        params: Dict[str, Any] = dict(cfg.get("params") or {})

        # Inject runtime values only for keys whose YAML value is null
        for key, value in runtime_params.items():
            if key in params and params[key] is None:
                params[key] = value

        # Apply offline overrides per-backend type
        if force_offline:
            cls_name = target.rsplit(".", 1)[-1]
            if cls_name == "WandbBackend":
                logger.INFO(f"build_backends: skipping WandbBackend in offline/airgapped environment")
                continue
            elif cls_name == "MLflowBackend":
                params.setdefault("offline", True)

        cls = get_obj_from_str(target)
        backends.append(cls(**params))

    return backends


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
    backends: Optional[List[LoggerBackend]] = None,
    run_name: Optional[str] = None,
    progress_type: Literal["tqdm", "table"] = "tqdm",
    save_csv: bool = True,
    save_batch_csv: bool = False,
    save_epoch_csv: bool = True,
    log_batch_tensorboard: bool = False,
    resume_epoch: int = 0,
) -> TrainingLogger:
    """
    Create a :class:`TrainingLogger` for supervised training (with validation).

    When *backends* is omitted a single :class:`TensorBoardBackend` writing
    to ``log_dir / run_name`` is created automatically.
    """
    actual_dir = os.path.join(log_dir, run_name) if run_name else log_dir
    if backends is None:
        backends = [TensorBoardBackend(actual_dir)]
    return TrainingLogger(
        log_dir=actual_dir,
        epochs=epochs,
        backends=backends,
        progress_type=progress_type,
        use_validation=True,
        save_csv=save_csv,
        save_batch_csv=save_batch_csv,
        save_epoch_csv=save_epoch_csv,
        log_batch_scalars=log_batch_tensorboard,
        resume_epoch=resume_epoch,
    )


def create_self_supervised_logger(
    log_dir: str,
    epochs: int,
    backends: Optional[List[LoggerBackend]] = None,
    run_name: Optional[str] = None,
    progress_type: Literal["tqdm", "table"] = "tqdm",
    save_csv: bool = True,
    save_batch_csv: bool = False,
    save_epoch_csv: bool = True,
    log_batch_tensorboard: bool = False,
    resume_epoch: int = 0,
) -> TrainingLogger:
    """
    Create a :class:`TrainingLogger` for self-supervised training (no validation).

    When *backends* is omitted a single :class:`TensorBoardBackend` writing
    to ``log_dir / run_name`` is created automatically.
    """
    actual_dir = os.path.join(log_dir, run_name) if run_name else log_dir
    if backends is None:
        backends = [TensorBoardBackend(actual_dir)]
    return TrainingLogger(
        log_dir=actual_dir,
        epochs=epochs,
        backends=backends,
        progress_type=progress_type,
        use_validation=False,
        save_csv=save_csv,
        save_batch_csv=save_batch_csv,
        save_epoch_csv=save_epoch_csv,
        log_batch_scalars=log_batch_tensorboard,
        resume_epoch=resume_epoch,
    )


# ------------------------------------------------------------------ #
# Test                                                                 #
# ------------------------------------------------------------------ #

def test_logger(progress_type: str = "tqdm", use_validation: bool = True):
    import tempfile, time
    mode_str = "with validation" if use_validation else "no validation (self-supervised)"
    _test_logger = Logger(__name__)
    _test_logger.INFO("=" * 70)
    _test_logger.INFO(f"Testing TrainingLogger  progress_type='{progress_type}'  {mode_str}")
    _test_logger.INFO("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(
            log_dir=tmpdir,
            epochs=5,
            backends=[TensorBoardBackend(tmpdir)],
            run_name=f"test_{progress_type}",
            progress_type=progress_type,
            use_validation=use_validation,
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

    _test_logger.INFO(f"Test done — logs: {tmpdir}")


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