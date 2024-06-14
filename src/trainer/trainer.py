import random
from pathlib import Path
from random import shuffle, randint

import PIL
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.trainer.base_trainer import BaseTrainer
from src.utils import inf_loop, MetricTracker
from src.logger.utils import plot_attention_to_buf


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metrics, optimizer, config, device, dataloaders,
                 lr_scheduler=None, len_epoch=None, skip_oom=True):
        super().__init__(model, criterion, metrics, optimizer, lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config

        self.log_step = 10

        self._setup_loaders(dataloaders, len_epoch=len_epoch)

        self.train_metrics = MetricTracker("loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer)
        self.evaluation_metrics = MetricTracker("loss", *[m.name for m in self.metrics], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
  
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            try:
                batch = self.process_batch(batch, is_train=True, metrics=self.train_metrics)

            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self._free_memory
                    continue
                else:
                    raise e
                
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug("Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(batch_idx), batch["loss"].item()))
                self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
                
                # TODO: add logs
                # self._log_predictions(**batch)
                self._log_attention(**batch)
                self._log_scalars(self.train_metrics)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        
        if is_train:
            self.optimizer.zero_grad()

        batch["fusion_score"] = self.model(**batch)
        loss_out = self.criterion(**batch)
        batch.update(loss_out)

        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.model.eval()
        with torch.no_grad():
            batch["x_prediction"] = self.model.predict(batch["x"])
            batch["y_prediction"] = self.model.predict(batch["y"])

        if is_train:
            self.model.train()

        for key in loss_out.keys():
            metrics.update(key, batch[key].item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, is_train=False, metrics=self.evaluation_metrics)

            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_attention(**batch)
            # TODO: add logs
            # self._log_predictions(**batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def _log_attention(self, batch):
        ind = randint(0, batch["x"].shape[-1] - 1)
        x = batch["x"][ind].unsqueeze(0)
        y = batch["y"][ind].unsqueeze(0)

        x_in = torch.cat([x, y])
        y_in = torch.cat([y, x])

        if batch["l_value"][ind] < 0.5:
            x_in, y_in = y_in, x_in

        small_batch = {
            "x": x_in, "y": y_in, "return_attention": True
        }

        with torch.no_grad():
            _, attention = self.model(**small_batch)
        
        for i, name in enumerate(["Clean vs Aug", "Aug vs Clean"]):
            image = PIL.Image.open(plot_attention_to_buf(attention[i].detach().cpu()))
            self.writer.add_image(name, ToTensor()(image))
        
    def _log_audio(self, audio_batch, name="audio"):
        audio = random.choice(audio_batch.cpu())
        self.writer.add_audio(name, audio, self.config["preprocessing"]["sr"])
    
    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _free_memory(self):
        self.logger.warning("OOM on batch. Skipping batch.")
        for p in self.model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()

    def _setup_loaders(self, dataloaders, len_epoch):
        use_inf_loop = len_epoch is not None
        self.train_dataloader = inf_loop(dataloaders["train"]) if use_inf_loop else dataloaders["train"]
        self.len_epoch = len_epoch if use_inf_loop else len(self.train_dataloader)
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p.grad is not None]
        parameters_stack = torch.stack([torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters])
        return torch.norm(parameters_stack, norm_type).item()

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(self.model.parameters(), self.config["trainer"]["grad_norm_clip"])

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["x", "y", "l_value"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch
