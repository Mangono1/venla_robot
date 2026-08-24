"""
VENLA V0.1
Training Engine V1

Fungsi:
- training Transformer
- AdamW optimizer
- CUDA / CPU
- AMP FP16
- gradient clipping
- checkpoint
- resume training
- training statistics
"""

import os
import time
import math

import torch
import torch.nn as nn


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRADIENT_CLIP = 1.0

DEFAULT_MAX_STEPS = 10000
DEFAULT_LOG_INTERVAL = 10
DEFAULT_CHECKPOINT_INTERVAL = 500

DEFAULT_BATCH_SIZE = 2

DEFAULT_USE_AMP = True


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# TRAINER
# ============================================================

class VENLATrainer:
    """
    Training engine VENLA V0.1.
    """

    def __init__(
        self,
        model,
        dataloader,
        device=None,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        gradient_clip=DEFAULT_GRADIENT_CLIP,
        max_steps=DEFAULT_MAX_STEPS,
        log_interval=DEFAULT_LOG_INTERVAL,
        checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
        checkpoint_dir=None,
        use_amp=DEFAULT_USE_AMP,
    ):

        self.model = model

        self.dataloader = dataloader

        self.device = (
            device
            if device is not None
            else get_device()
        )


        # ----------------------------------------------------
        # TRAINING CONFIG
        # ----------------------------------------------------

        self.learning_rate = float(
            learning_rate
        )

        self.weight_decay = float(
            weight_decay
        )

        self.gradient_clip = float(
            gradient_clip
        )

        self.max_steps = int(
            max_steps
        )

        self.log_interval = int(
            log_interval
        )

        self.checkpoint_interval = int(
            checkpoint_interval
        )


        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        self.checkpoint_dir = (
            checkpoint_dir
            or os.path.join(
                "checkpoints"
            )
        )

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True
        )


        # ----------------------------------------------------
        # AMP
        # ----------------------------------------------------

        self.use_amp = bool(
            use_amp
            and
            self.device.type == "cuda"
        )


        # ----------------------------------------------------
        # SCALER
        # ----------------------------------------------------

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )


        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.step = 0

        self.epoch = 0

        self.last_loss = None

        self.loss_history = []

        self.tokens_processed = 0

        self.start_time = None


        # ----------------------------------------------------
        # MODEL DEVICE
        # ----------------------------------------------------

        self.model.to(
            self.device
        )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    def get_learning_rate(self):

        return self.optimizer.param_groups[
            0
        ][
            "lr"
        ]


    def set_learning_rate(
        self,
        learning_rate,
    ):

        learning_rate = float(
            learning_rate
        )

        for group in self.optimizer.param_groups:

            group[
                "lr"
            ] = learning_rate


    # ========================================================
    # ZERO GRADIENT
    # ========================================================

    def zero_grad(self):

        self.optimizer.zero_grad(
            set_to_none=True
        )


    # ========================================================
    # TRAIN STEP
    # ========================================================

    def train_step(
        self,
        input_ids,
        targets,
    ):

        self.model.train()

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        targets = targets.to(
            self.device,
            non_blocking=True,
        )


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        with torch.autocast(
            device_type=self.device.type,
            dtype=(
                torch.float16
                if self.device.type == "cuda"
                else torch.float32
            ),
            enabled=self.use_amp,
        ):

            logits, loss = self.model(
                input_ids,
                targets,
            )


        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        self.scaler.scale(
            loss
        ).backward()


        # ----------------------------------------------------
        # GRADIENT CLIPPING
        # ----------------------------------------------------

        if self.gradient_clip > 0:

            self.scaler.unscale_(
                self.optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip,
            )


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        self.scaler.step(
            self.optimizer
        )

        self.scaler.update()

        self.zero_grad()


        return loss


    # ========================================================
    # CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        path=None,
    ):

        if path is None:

            path = os.path.join(
                self.checkpoint_dir,
                "latest.pt",
            )


        checkpoint = {
            "checkpoint_format":
                "VENLA_TRAINER_V1",

            "step":
                self.step,

            "epoch":
                self.epoch,

            "last_loss":
                self.last_loss,

            "tokens_processed":
                self.tokens_processed,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scaler_state_dict":
                self.scaler.state_dict(),

            "model_config":
                (
                    self.model.model_config()
                    if hasattr(
                        self.model,
                        "model_config"
                    )
                    else {}
                ),

        }


        torch.save(
            checkpoint,
            path,
        )


        return path


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def load_checkpoint(
        self,
        path,
    ):

        if not os.path.exists(path):

            raise FileNotFoundError(
                "Checkpoint tidak ditemukan: "
                + path
            )


        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer_state = checkpoint.get(
            "optimizer_state_dict"
        )

        if optimizer_state is not None:

            self.optimizer.load_state_dict(
                optimizer_state
            )


        # ----------------------------------------------------
        # SCALER
        # ----------------------------------------------------

        scaler_state = checkpoint.get(
            "scaler_state_dict"
        )

        if scaler_state is not None:

            self.scaler.load_state_dict(
                scaler_state
            )


        # ----------------------------------------------------
        # TRAINING STATE
        # ----------------------------------------------------

        self.step = int(
            checkpoint.get(
                "step",
                0,
            )
        )

        self.epoch = int(
            checkpoint.get(
                "epoch",
                0,
            )
        )

        self.last_loss = checkpoint.get(
            "last_loss"
        )

        self.tokens_processed = int(
            checkpoint.get(
                "tokens_processed",
                0,
            )
        )


        return checkpoint


    # ========================================================
    # LOGGING
    # ========================================================

    def print_progress(
        self,
        loss,
    ):

        elapsed = (
            time.time()
            - self.start_time
            if self.start_time is not None
            else 0
        )


        steps_per_second = (
            self.step / elapsed
            if elapsed > 0
            else 0
        )


        print(
            f"step={self.step:6d} "
            f"epoch={self.epoch:4d} "
            f"loss={float(loss):.5f} "
            f"lr={self.get_learning_rate():.6g} "
            f"speed={steps_per_second:.2f} step/s"
        )


    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        max_steps=None,
    ):

        if max_steps is None:

            max_steps = self.max_steps

        max_steps = int(
            max_steps
        )


        self.start_time = time.time()


        print("=" * 60)
        print("VENLA V0.1 - TRAINING")
        print("=" * 60)

        print()

        print(
            "Device:",
            self.device
        )

        print(
            "AMP:",
            self.use_amp
        )

        print(
            "Learning rate:",
            self.learning_rate
        )

        print(
            "Weight decay:",
            self.weight_decay
        )

        print(
            "Max steps:",
            max_steps
        )

        print()


        # ----------------------------------------------------
        # TRAINING LOOP
        # ----------------------------------------------------

        while self.step < max_steps:

            self.epoch += 1


            for batch in self.dataloader:

                if self.step >= max_steps:

                    break


                input_ids, targets = batch


                loss = self.train_step(
                    input_ids,
                    targets,
                )


                self.step += 1


                self.last_loss = float(
                    loss.detach().cpu()
                )


                self.loss_history.append(
                    self.last_loss
                )


                self.tokens_processed += (
                    input_ids.numel()
                )


                # ------------------------------------------------
                # LOG
                # ------------------------------------------------

                if (
                    self.step
                    % self.log_interval
                    == 0
                ):

                    self.print_progress(
                        loss
                    )


                # ------------------------------------------------
                # CHECKPOINT
                # ------------------------------------------------

                if (
                    self.step
                    % self.checkpoint_interval
                    == 0
                ):

                    checkpoint_path = (
                        os.path.join(
                            self.checkpoint_dir,
                            f"step_{self.step}.pt",
                        )
                    )

                    self.save_checkpoint(
                        checkpoint_path
                    )

                    self.save_checkpoint(
                        os.path.join(
                            self.checkpoint_dir,
                            "latest.pt",
                        )
                    )


                    print(
                        "Checkpoint:",
                        checkpoint_path
                    )


        # ----------------------------------------------------
        # FINAL CHECKPOINT
        # ----------------------------------------------------

        final_path = os.path.join(
            self.checkpoint_dir,
            "final.pt",
        )

        self.save_checkpoint(
            final_path
        )


        elapsed = (
            time.time()
            - self.start_time
        )


        print()

        print("=" * 60)
        print("TRAINING SELESAI")
        print("=" * 60)

        print()

        print(
            "Steps:",
            self.step
        )

        print(
            "Epoch:",
            self.epoch
        )

        print(
            "Final loss:",
            self.last_loss
        )

        print(
            "Tokens processed:",
            self.tokens_processed
        )

        print(
            "Elapsed:",
            f"{elapsed:.2f} seconds"
        )

        print(
            "Final checkpoint:",
            final_path
        )

        print()

        return {
            "step":
                self.step,

            "epoch":
                self.epoch,

            "loss":
                self.last_loss,

            "tokens_processed":
                self.tokens_processed,

            "elapsed_seconds":
                elapsed,

            "checkpoint":
                final_path,
        }


# ============================================================
# TRAINING SUMMARY
# ============================================================

def training_summary(
    trainer,
):

    return {
        "step":
            trainer.step,

        "epoch":
            trainer.epoch,

        "loss":
            trainer.last_loss,

        "tokens_processed":
            trainer.tokens_processed,

        "learning_rate":
            trainer.get_learning_rate(),

        "device":
            str(
                trainer.device
            ),

        "amp":
            trainer.use_amp,
    }


# ============================================================
# TEST TRAINING ENGINE
# ============================================================

def test_trainer():

    print("=" * 60)
    print("VENLA V0.1 - TRAINER TEST")
    print("=" * 60)

    print()

    from .model import VENLA
    from .tokenizer import VENLATokenizer
    from .dataset import (
        VENLADatasetEngine,
    )


    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = VENLATokenizer(
        vocab_size=32768
    )


    # --------------------------------------------------------
    # SAMPLE TEXT
    # --------------------------------------------------------

    text = (
        "VENLA adalah model bahasa "
        "yang sedang belajar. "
        "VENLA membaca teks Indonesia "
        "dan memprediksi token berikutnya. "
    ) * 100


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    engine = VENLADatasetEngine(
        tokenizer=tokenizer,
        context_length=512,
    )

    tokens = engine.tokenize_text(
        text
    )

    engine.build_dataset(
        tokens
    )

    dataloader = engine.build_dataloader(
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    device = get_device()

    model = VENLA().to(
        device
    )


    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = VENLATrainer(
        model=model,
        dataloader=dataloader,
        device=device,
        learning_rate=3e-4,
        weight_decay=0.1,
        gradient_clip=1.0,
        max_steps=2,
        log_interval=1,
        checkpoint_interval=2,
        checkpoint_dir=os.path.join(
            "checkpoints",
            "test",
        ),
        use_amp=True,
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    result = trainer.train(
        max_steps=2
    )


    print()

    print(
        "Training result:"
    )

    print(
        result
    )

    print()

    print(
        "✅ TRAINER ENGINE TEST SELESAI"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_trainer()
