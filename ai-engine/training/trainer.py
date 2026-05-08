"""
trainer.py — Fine-tuning loop for LaMa inpainter.

Training phases:
  Phase 1 (epochs 1-10):   Freeze encoder, train decoder only.
                            Fast convergence — decoder adapts to hole shapes.
  Phase 2 (epochs 11-20):  Unfreeze encoder, full fine-tuning at lower LR.
                            Refines global structure for your specific domain.

Mixed precision training (torch.cuda.amp) is used throughout —
halves VRAM usage and speeds up training ~1.5-2x on modern GPUs.

Run:
  python training/trainer.py --davis_root /path/to/davis --device cuda
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from synthesis.inpainter import LaMaInpainter
from training.losses import InpainterLoss, PatchDiscriminator, AdversarialLoss
from training.dataset import make_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class InpainterTrainer:
    """
    Two-phase fine-tuning trainer for LaMa inpainter.

    Args:
        model:           LaMaInpainter instance.
        davis_root:      Path to DAVIS dataset. None = synthetic data.
        save_dir:        Where to save checkpoints and logs.
        device:          'cuda' or 'cpu'.
        batch_size:      Training batch size. 8 for 8GB VRAM.
        lr_decoder:      Learning rate for phase 1 (decoder only).
        lr_full:         Learning rate for phase 2 (full model).
        use_adversarial: Whether to train PatchGAN discriminator.
    """

    def __init__(
      self,
      model: LaMaInpainter,
      davis_root: str = None,             # type: ignore[assignment]
      save_dir: str = 'checkpoints',
      device: str = 'cpu',
      batch_size: int = 8,
      lr_decoder: float = 2e-4,
      lr_full: float = 5e-5,
      use_adversarial: bool = True,
      phase1_epochs: int = 10,
      phase2_epochs: int = 10,
    ):
      self.model      = model.to(device)
      self.device     = device
      self.save_dir   = Path(save_dir)
      self.save_dir.mkdir(parents=True, exist_ok=True)
      self.phase1_epochs = phase1_epochs
      self.phase2_epochs = phase2_epochs

      # Dataloaders
      self.train_loader = make_dataloader(
        davis_root, split='train',
        batch_size=batch_size, num_workers=4,
      )
      self.val_loader = make_dataloader(
        davis_root, split='val',
        batch_size=batch_size, num_workers=2,
      )

      # Losses
      self.criterion = InpainterLoss(
        use_adversarial=use_adversarial,
        device=device,
      ).to(device)

      # Discriminator (optional)
      self.use_adv = use_adversarial
      if use_adversarial:
        self.discriminator = PatchDiscriminator().to(device)
        self.adv_loss      = AdversarialLoss()
        self.opt_d = optim.Adam(
          self.discriminator.parameters(),
          lr=lr_decoder, betas=(0.5, 0.999),
        )

      self.lr_decoder = lr_decoder
      self.lr_full    = lr_full

      # FIX: Phase 1 optimizer is built explicitly over decoder parameters
      # only. Building it from model.decoder.parameters() directly avoids
      # accidentally including encoder params before freeze_encoder() is
      # called (encoder params are requires_grad=True at __init__ time).
      self.opt_g = optim.Adam(
        self.model.decoder.parameters(),
        lr=lr_decoder, betas=(0.9, 0.999),
      )

      # Mixed precision scalers
      self.scaler_g = GradScaler(enabled=(device == 'cuda'))
      self.scaler_d = GradScaler(enabled=(device == 'cuda'))

      # TensorBoard
      self.writer = SummaryWriter(self.save_dir / 'logs')
      self.global_step = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
      log.info("=== Phase 1: Decoder fine-tuning (encoder frozen) ===")
      self.model.freeze_encoder()
      self._run_phase(self.phase1_epochs, phase=1)

      log.info("=== Phase 2: Full model fine-tuning ===")
      self.model.unfreeze_encoder()
      # Switch optimiser to cover all parameters at a lower LR
      self.opt_g = optim.Adam(
        self.model.parameters(),
        lr=self.lr_full, betas=(0.9, 0.999),
      )
      self._run_phase(self.phase2_epochs, phase=2)

      log.info("Training complete.")
      self.writer.close()

    def _run_phase(self, n_epochs: int, phase: int) -> None:
      for epoch in range(1, n_epochs + 1):
        train_losses = self._train_epoch(epoch, phase)
        val_losses   = self._val_epoch(epoch)

        for k, v in train_losses.items():
          self.writer.add_scalar(f'train/{k}', v, epoch + (phase - 1) * self.phase1_epochs)
        for k, v in val_losses.items():
          self.writer.add_scalar(f'val/{k}', v, epoch + (phase - 1) * self.phase1_epochs)

        log.info(
          f"Phase {phase} | Epoch {epoch}/{n_epochs} | "
          f"train_total={train_losses['total']:.4f} | "
          f"val_total={val_losses['total']:.4f}"
        )

        self._save_checkpoint(epoch, phase, val_losses['total'])

    def _train_epoch(self, epoch: int, phase: int) -> dict:
      self.model.train()
      accum: dict[str, float] = {}

      for batch in self.train_loader:
        masked = batch['masked_image'].to(self.device)
        mask   = batch['hole_mask'].to(self.device)
        target = batch['target'].to(self.device)

        # --- Discriminator step ---
        if self.use_adv:
          # FIX: autocast requires device_type as an explicit kwarg (PyTorch 2.x).
          with autocast(device_type=self.device, enabled=(self.device == 'cuda')):
            with torch.no_grad():
              fake = self.model(masked, mask)
            real_scores = self.discriminator(target, mask)
            fake_scores = self.discriminator(fake.detach(), mask)
            d_loss      = self.adv_loss.discriminator_loss(real_scores, fake_scores)

          self.opt_d.zero_grad()
          self.scaler_d.scale(d_loss).backward()
          self.scaler_d.step(self.opt_d)
          self.scaler_d.update()

        # --- Generator step ---
        # FIX: was autocast(enabled=(True, device_type=self.device)) — a
        # tuple was being passed to `enabled`, which is a boolean argument.
        with autocast(device_type=self.device, enabled=(self.device == 'cuda')):
          output      = self.model(masked, mask)
          fake_scores = self.discriminator(output, mask) if self.use_adv else None
          losses      = self.criterion(output, target, mask, fake_scores)

        self.opt_g.zero_grad()
        self.scaler_g.scale(losses['total']).backward()

        # Gradient clipping — prevents explosion in early fine-tuning
        self.scaler_g.unscale_(self.opt_g)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        for k, v in losses.items():
          accum[k] = accum.get(k, 0.0) + v.item()

        self.global_step += 1

      n = len(self.train_loader)
      return {k: v / n for k, v in accum.items()}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
      self.model.eval()
      accum: dict[str, float] = {}
      first_batch_logged = False

      for batch in self.val_loader:
        masked = batch['masked_image'].to(self.device)
        mask   = batch['hole_mask'].to(self.device)
        target = batch['target'].to(self.device)

        # FIX: was autocast(enabled=(...)) with a type: ignore suppressing
        # a missing device_type arg. Corrected to explicit device_type kwarg.
        with autocast(device_type=self.device, enabled=(self.device == 'cuda')):
          output = self.model(masked, mask)
          losses = self.criterion(output, target, mask)

        for k, v in losses.items():
          accum[k] = accum.get(k, 0.0) + v.item()

        if not first_batch_logged:
          self._log_images(masked, mask, output, target, epoch)
          first_batch_logged = True

      n = len(self.val_loader)
      return {k: v / n for k, v in accum.items()}

    def _log_images(self, masked, mask, output, target, epoch):
      """Log comparison images to TensorBoard."""
      from torchvision.utils import make_grid
      grid = make_grid(
        torch.cat([masked[:4], output[:4], target[:4]], dim=0),
        nrow=4, normalize=False,
      )
      self.writer.add_image('val/masked_output_target', grid, epoch)

    def _save_checkpoint(self, epoch: int, phase: int, val_loss: float) -> None:
      path = self.save_dir / f'phase{phase}_epoch{epoch:03d}_val{val_loss:.4f}.pth'
      torch.save({
        'epoch':       epoch,
        'phase':       phase,
        'model':       self.model.state_dict(),
        'opt_g':       self.opt_g.state_dict(),
        'val_loss':    val_loss,
        'global_step': self.global_step,
      }, path)
      log.info(f"Saved checkpoint: {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
  p = argparse.ArgumentParser(description='Fine-tune LaMa inpainter')
  p.add_argument('--davis_root',     type=str, default=None)
  p.add_argument('--pretrained',     type=str, default=None,
                  help='Path to LaMa pretrained weights')
  p.add_argument('--save_dir',       type=str, default='checkpoints/inpainter')
  p.add_argument('--device',         type=str, default='cuda')
  p.add_argument('--batch_size',     type=int, default=8)
  p.add_argument('--phase1_epochs',  type=int, default=10)
  p.add_argument('--phase2_epochs',  type=int, default=10)
  p.add_argument('--no_adversarial', action='store_true')
  return p.parse_args()


if __name__ == '__main__':
  args = parse_args()

  device = args.device if torch.cuda.is_available() else 'cpu'
  log.info(f"Device: {device}")

  model = LaMaInpainter(base_channels=64)

  if args.pretrained:
    model.load_pretrained(args.pretrained)
  else:
    log.info("No pretrained weights — training from random init (not recommended)")

  params = model.count_parameters()
  log.info(f"Model parameters: {params}")

  trainer = InpainterTrainer(
    model=model,
    davis_root=args.davis_root,
    save_dir=args.save_dir,
    device=device,
    batch_size=args.batch_size,
    use_adversarial=not args.no_adversarial,
    phase1_epochs=args.phase1_epochs,
    phase2_epochs=args.phase2_epochs,
  )

  trainer.train()