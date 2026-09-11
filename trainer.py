"""A small, dependency-light training loop shared by all three tasks.

Each task supplies: a model, a `loss_fn(outputs, batch) -> dict`, a
`metrics_fn(outputs, batch) -> dict`, and a `forward_fn(model, batch) -> dict`
that knows how to call the model given that task's batch dict. This keeps
one Trainer implementation instead of three near-duplicate loops.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import ExperimentLogger, get_device


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Returns True if `score` is the new best."""
        if self.best_score is None:
            self.best_score = score
            return True

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics_fn: Callable,
        forward_fn: Callable,
        run_dir: str,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        grad_accum_steps: int = 1,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 10,
        monitor_metric: str = "total_loss",
        monitor_mode: str = "min",
        device: str = "auto",
        use_amp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.forward_fn = forward_fn
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip_norm = grad_clip_norm
        self.monitor_metric = monitor_metric

        self.device = get_device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=monitor_mode, factor=0.5, patience=max(1, early_stopping_patience // 2)
        )
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode=monitor_mode)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger(self.run_dir, name=self.run_dir.name)
        self.best_state: Optional[dict] = None

    def _move_batch(self, batch: Dict) -> Dict:
        moved = {}
        for k, v in batch.items():
            moved[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return moved

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {}
        n_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"epoch {epoch} [train]", leave=False)
        for step, batch in enumerate(pbar):
            batch = self._move_batch(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.forward_fn(self.model, batch)
                losses = self.loss_fn(outputs, batch)
                loss = losses["total_loss"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.item())
            n_batches += 1
            pbar.set_postfix({k: f"{v / n_batches:.4f}" for k, v in running.items()})

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict[str, float]:
        self.model.eval()
        running_losses: Dict[str, float] = {}
        running_metrics: Dict[str, float] = {}
        n_batches = 0

        for batch in tqdm(loader, desc=f"[{split}]", leave=False):
            batch = self._move_batch(batch)
            outputs = self.forward_fn(self.model, batch)
            losses = self.loss_fn(outputs, batch)
            metrics = self.metrics_fn(outputs, batch)

            for k, v in losses.items():
                running_losses[k] = running_losses.get(k, 0.0) + float(v.item())
            for k, v in metrics.items():
                if v == v:  # skip NaN
                    running_metrics[k] = running_metrics.get(k, 0.0) + float(v)
            n_batches += 1

        out = {k: v / max(n_batches, 1) for k, v in running_losses.items()}
        out.update({k: v / max(n_batches, 1) for k, v in running_metrics.items()})
        return out

    def fit(self) -> Dict[str, float]:
        best_val_metrics: Dict[str, float] = {}
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, split="val")

            log_row = {f"train_{k}": round(v, 5) for k, v in train_metrics.items()}
            log_row.update({f"val_{k}": round(v, 5) for k, v in val_metrics.items()})
            self.logger.log(epoch, log_row)

            monitored = val_metrics.get(self.monitor_metric, val_metrics.get("total_loss"))
            self.scheduler.step(monitored)

            is_best = self.early_stopping.step(monitored)
            if is_best:
                best_val_metrics = val_metrics
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.save_checkpoint("best.pt")
            self.save_checkpoint("last.pt")

            if self.early_stopping.should_stop:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for "
                    f"{self.early_stopping.patience} epochs)."
                )
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return best_val_metrics

    def save_checkpoint(self, filename: str) -> None:
        torch.save(
            {"model_state_dict": self.model.state_dict()},
            self.run_dir / filename,
        )
