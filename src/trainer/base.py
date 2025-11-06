from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import yaml
from numpy import inf
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.utils import inf_loop
from src.logger.base import BaseWriter
from src.loss.base import BaseLoss
from src.metrics.base import BaseMetric
from src.metrics.tracker import MetricTracker
from src.model.base import BaseModel
from src.utils.io import get_root


class BaseTrainer:
    """
    Base class for all trainers.

    Provides common training infrastructure including epoch management,
    metric tracking, checkpointing, and evaluation logic.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: str,
        dataloaders: dict[str, DataLoader],
        model: BaseModel,
        criterion: BaseLoss,
        metrics: dict[str, list[BaseMetric]],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        logger: Logger,
        writer: BaseWriter,
        batch_transforms: dict[str, dict[str, nn.Sequential]],
        skip_oom: bool = True,
        epoch_len: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        cfg : DictConfig
            Experiment config containing training config.
        device : str
            Device for tensors and model.
        dataloaders : dict[str, DataLoader]
            Dataloaders for different sets of data.
        model : BaseModel
            PyTorch model.
        criterion : BaseLoss
            Loss function for model training.
        metrics : dict[str, list[BaseMetric]]
            Dict with the definition of metrics for training
            (metrics['train']) and inference (metrics['inference']). Each
            metric is an instance of src.metrics.BaseMetric.
        optimizer : Optimizer
            Optimizer for the model.
        lr_scheduler : LRScheduler
            Learning rate scheduler for the optimizer.
        logger : Logger
            Logger that logs output.
        writer : BaseWriter
            Experiment tracker.
        batch_transforms : dict[str, dict[str, nn.Sequential]]
            Transforms that should be applied on the whole batch. Depend on the
            tensor name.
        skip_oom : bool, optional
            Skip batches with the OutOfMemory error, by default True.
        epoch_len : int or None, optional
            Number of steps in each epoch for iteration-based training.
            If None, use epoch-based training (len(dataloader)), by default None.
        """
        self.is_train = True

        self.cfg = cfg
        self.cfg_trainer = self.cfg.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = self.cfg_trainer.get('log_step', 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        if epoch_len is None:
            self.train_dataloader = dataloaders['train']
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(dataloaders['train'])
            self.epoch_len = epoch_len
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.num_epochs = self.cfg_trainer.num_epochs

        # configuration to monitor model performance and save best
        self.save_period = self.cfg_trainer.save_period  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get('monitor', 'off')  # format: "mnt_mode mnt_metric"

        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.cfg_trainer.get('early_stop', inf)
            if self.early_stop is None or self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics_tracker = MetricTracker(
            *self.criterion.loss_names,
            'grad_norm',
            *[m.name for m in self.metrics['train']],
        )
        self.eval_metrics_tracker = MetricTracker(
            *self.criterion.loss_names,
            *[m.name for m in self.metrics['inference']],
        )

        # define checkpoint dir and init everything if required
        self.checkpoint_dir: Path = get_root() / self.cfg_trainer.checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        if self.cfg_trainer.get('resume_from') is not None:
            resume_path = self.checkpoint_dir / self.cfg_trainer.resume_from
            self._resume_checkpoint(resume_path)

        if self.cfg_trainer.get('from_pretrained') is not None:
            self._from_pretrained(self.cfg_trainer.from_pretrained)

    def train(self) -> None:
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self._log('Saving model on keyboard interrupt')
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self) -> None:
        """
        Full training logic.

        Trains model for multiple epochs, evaluates it on non-train partitions,
        and monitors the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {'epoch': epoch}
            logs.update(result)

            # print logged information to the screen
            self._log(f'\nMetrics:\n{yaml.dump(logs)}')

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best or epoch == self.num_epochs:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:  # early_stop
                break

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        logs : dict[str, float]
            Logs that contain the average loss and metric in this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics_tracker.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar('epoch', epoch)

        pbar = tqdm(self.train_dataloader, desc=f'Train Epoch {epoch}', total=self.epoch_len)
        for batch_idx, batch in enumerate(pbar):
            try:
                batch = self.process_batch(
                    batch,
                    metric_tracker=self.train_metrics_tracker,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning('OOM on batch. Skipping batch.')
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            grad_norm = self._get_grad_norm()
            pbar.set_postfix({'loss': batch['loss'].item(), 'grad_norm': grad_norm})
            self.train_metrics_tracker.update('grad_norm', grad_norm)

            # log current results
            if batch_idx > 0 and batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.writer.add_scalar('lr', self.lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics_tracker)
                self._log_batch(batch_idx, batch, epoch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics_tracker.result()
                self.train_metrics_tracker.reset()

            if batch_idx + 1 > self.epoch_len:
                break

        logs = last_train_metrics
        logs = {f'train_{name}': value for name, value in logs.items()}

        # Run val/test
        for part, dataloader in self.eval_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f'{part}_{name}': value for name, value in val_logs.items()})

        return logs

    @torch.no_grad()
    def _evaluation_epoch(
        self,
        epoch: int,
        part: Literal['train', 'val', 'test'],
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """
        Evaluate model on the partition after training for an epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        part : {'train', 'val', 'test'}
            Partition to evaluate on.
        dataloader : DataLoader
            Dataloader for the partition.

        Returns
        -------
        logs : dict[str, float]
            Logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.eval_metrics_tracker.reset()

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc=f'{part.capitalize()} Epoch {epoch}',
            total=len(dataloader),
        ):
            batch = self.process_batch(
                batch,
                metric_tracker=self.eval_metrics_tracker,
                part=part,
            )

        self.writer.set_step(epoch * self.epoch_len, part)

        batch['sample'] = self.model.sample(
            num_samples=batch['x'].shape[0],
            device=batch['x'].device,
        )

        self._log_scalars(self.eval_metrics_tracker)
        self._log_batch(batch_idx, batch, epoch)
        return self.eval_metrics_tracker.result()

    def process_batch(
        self,
        batch: dict[str, Any],
        metric_tracker: MetricTracker,
        part: Literal['train', 'val', 'test'] = 'train',
    ) -> dict[str, Any]:
        raise NotImplementedError(f'{type(self).__name__} must implement process_batch method')

    def _log_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        epoch: int,
    ):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch.
        batch : dict[str, Any]
            Dict-based batch after going through the 'process_batch' function.
        epoch : int
            Current epoch number.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement _log_batch method')

    def _monitor_performance(
        self,
        logs: dict[str, float],
        not_improved_count: int,
    ) -> tuple[bool, bool, int]:
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Parameters
        ----------
        logs : dict[str, float]
            Logs after training and evaluating the model for an epoch.
        not_improved_count : int
            The current number of epochs without improvement.

        Returns
        -------
        best : bool
            If True, the monitored metric has improved.
        stop_process : bool
            If True, stop the process (early stopping).
            The metric did not improve for too many epochs.
        not_improved_count : int
            Updated number of epochs without improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == 'min':
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == 'max':
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self._log(
                    message=f"Warning: Metric '{self.mnt_metric}' is not found. "
                    'Model performance monitoring is disabled.',
                    message_type='WARNING',
                )
                self.mnt_mode = 'off'
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self._log(
                    f"Validation performance didn't improve for {self.early_stop} epochs. Training stops."
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Move all necessary tensors to the device.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.

        Returns
        -------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader
            with some of the tensors on the device.
        """
        for key in self.cfg_trainer.device_tensors:
            batch[key] = batch[key].to(self.device)
        return batch

    def _transform_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.

        Returns
        -------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader
            (possibly transformed via batch transform).
        """
        transform_type = 'train' if self.is_train else 'inference'
        transforms = self.batch_transforms.get(transform_type)

        if transforms is None:
            return batch

        for transform_name, transform_fn in transforms.items():  # ty: ignore[ call-non-callable]
            batch[transform_name] = transform_fn(batch[transform_name])
        return batch

    def _clip_grad_norm(self) -> None:
        """
        Clips the gradient norm by the value defined in cfg.trainer.max_grad_norm.
        """
        if self.cfg_trainer.get('max_grad_norm', None) is not None:
            clip_grad_norm_(self.model.parameters(), self.cfg_trainer.max_grad_norm)

    @torch.no_grad()
    def _get_grad_norm(self, norm_type: float | str | None = 2) -> float:
        """
        Calculates the gradient norm for logging.

        Parameters
        ----------
        norm_type : float or str or None, optional
            The order of the norm, by default 2.

        Returns
        -------
        total_norm : float
            The calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker) -> None:
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Parameters
        ----------
        metric_tracker : MetricTracker
            Calculated metrics.
        """
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f'{metric_name}', metric_tracker[metric_name])

    def _save_checkpoint(
        self,
        epoch: int,
        save_best: bool = False,
        only_best: bool = False,
    ) -> None:
        """
        Save the checkpoints.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        save_best : bool, optional
            If True, rename the saved checkpoint to 'model_best.pth', by default False.
        only_best : bool, optional
            If True and the checkpoint is the best, save it only as
            'model_best.pth' (do not duplicate the checkpoint as
            checkpoint-epoch-Epoch-Number.pth), by default False.
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'monitor_best': f'{self.mnt_metric}: {self.mnt_best}',
            'cfg': self.cfg,
        }
        filename = str(self.checkpoint_dir / f'checkpoint-epoch-{epoch}.pth')
        if not (only_best and save_best):
            torch.save(state, filename)
            self._log(f'Saving checkpoint: {filename} ...')
            self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self._log('Saving current best: model_best.pth ...')

    def _resume_checkpoint(self, resume_path: str) -> None:
        """
        Resume from a saved checkpoint (in case of server crash, etc.).

        The function loads state dicts for everything, including model,
        optimizers, etc.

        Parameters
        ----------
        resume_path : str
            Path to the checkpoint to be resumed.

        Notes
        -----
        The checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').
        """
        resume_path = str(resume_path)
        self._log(f'Loading checkpoint: {resume_path} ...')
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.cfg['model']:
            self._log(
                message='Warning: Architecture configuration given in the config file is different from that '
                'of the checkpoint. This may yield an exception when state_dict is loaded.',
                message_type='WARNING',
            )
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint['config']['optimizer'] != self.cfg['optimizer']
            or checkpoint['config']['lr_scheduler'] != self.cfg['lr_scheduler']
        ):
            self._log(
                message='Warning: Optimizer or lr_scheduler given in the config file is different '
                'from that of the checkpoint. Optimizer and scheduler parameters '
                'are not resumed.',
                message_type='WARNING',
            )
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self._log(f'Checkpoint loaded. Resume training from epoch {self.start_epoch}')

    def _from_pretrained(self, pretrained_path: str) -> None:
        """
        Init model with weights from pretrained pth file.

        Parameters
        ----------
        pretrained_path : str
            Path to the model state dict.

        Notes
        -----
        'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.
        """
        pretrained_path = str(pretrained_path)
        self._log(f'Loading model weights from: {pretrained_path} ...')
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get('state_dict') is not None:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def _log(
        self,
        message: str,
        message_type: Literal['INFO', 'WARNING', 'DEBUG'] = 'INFO',
    ) -> None:
        """
        Log a message using the configured logger.

        Parameters
        ----------
        message : str
            Message to log.
        message_type : {'INFO', 'WARNING', 'DEBUG'}, optional
            Type of the log message, by default 'INFO'.
        """
        message = f'{type(self).__name__} {message}'
        if self.logger is not None:
            match message_type:
                case 'INFO':
                    self.logger.info(message)
                case 'DEBUG':
                    self.logger.debug(message)
                case 'WARNING':
                    self.logger.warning(message)
                case _:
                    self.logger.info(message)
        else:
            print(f'{datetime.now()} {message_type}: {message}')
