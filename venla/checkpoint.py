"""
VENLA V0.1
Checkpoint Manager

Menangani:
- save model
- load model
- optimizer state
- scheduler state
- training state
- metadata
- konfigurasi model
"""

import json
import os
import tempfile
from datetime import datetime, timezone

import torch


# ============================================================
# CHECKPOINT VERSION
# ============================================================

CHECKPOINT_FORMAT = "VENLA_CHECKPOINT_V1"


# ============================================================
# UTILITY
# ============================================================

def utc_now():
    """
    Return current UTC timestamp.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):
    """
    Count all model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# CHECKPOINT METADATA
# ============================================================

def build_metadata(
    model,
    step=0,
    epoch=0,
    loss=None,
    extra=None,
):
    """
    Build checkpoint metadata.
    """

    metadata = {
        "checkpoint_format":
            CHECKPOINT_FORMAT,

        "created_at":
            utc_now(),

        "model_name":
            "VENLA",

        "model_version":
            "V0.1",

        "parameters":
            count_parameters(model),

        "trainable_parameters":
            count_trainable_parameters(model),

        "step":
            int(step),

        "epoch":
            int(epoch),

        "loss":
            None
            if loss is None
            else float(loss),
    }


    # --------------------------------------------------------
    # MODEL CONFIG
    # --------------------------------------------------------

    if hasattr(
        model,
        "model_config"
    ):

        try:

            metadata[
                "model_config"
            ] = model.model_config()

        except Exception:

            metadata[
                "model_config"
            ] = {}


    # --------------------------------------------------------
    # EXTRA METADATA
    # --------------------------------------------------------

    if extra is not None:

        metadata[
            "extra"
        ] = dict(extra)


    return metadata


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    step=0,
    epoch=0,
    loss=None,
    extra=None,
):
    """
    Save complete VENLA checkpoint.

    Parameters
    ----------
    path:
        Destination .pt file.

    model:
        VENLA model.

    optimizer:
        Optional optimizer.

    scheduler:
        Optional scheduler.

    step:
        Current training step.

    epoch:
        Current epoch.

    loss:
        Current loss.

    extra:
        Additional metadata.
    """

    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )


    metadata = build_metadata(
        model=model,
        step=step,
        epoch=epoch,
        loss=loss,
        extra=extra,
    )


    checkpoint = {
        "checkpoint_format":
            CHECKPOINT_FORMAT,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            None
            if optimizer is None
            else optimizer.state_dict(),

        "scheduler_state_dict":
            None
            if scheduler is None
            else scheduler.state_dict(),

        "step":
            int(step),

        "epoch":
            int(epoch),

        "loss":
            None
            if loss is None
            else float(loss),

        "metadata":
            metadata,
    }


    # ========================================================
    # ATOMIC SAVE
    # ========================================================

    fd, temporary_path = tempfile.mkstemp(
        suffix=".tmp",
        dir=directory,
    )

    os.close(fd)


    try:

        torch.save(
            checkpoint,
            temporary_path,
        )

        os.replace(
            temporary_path,
            path,
        )

    except Exception:

        if os.path.exists(
            temporary_path
        ):

            os.remove(
                temporary_path
            )

        raise


    return metadata


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    map_location=None,
    strict=True,
):
    """
    Load complete VENLA checkpoint.

    Returns
    -------
    dict
        Training state and metadata.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {path}"
        )


    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )


    # ========================================================
    # FORMAT VALIDATION
    # ========================================================

    checkpoint_format = checkpoint.get(
        "checkpoint_format"
    )


    if checkpoint_format != CHECKPOINT_FORMAT:

        raise RuntimeError(
            "Format checkpoint tidak dikenal: "
            + str(checkpoint_format)
        )


    # ========================================================
    # MODEL
    # ========================================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=strict,
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )


    if (
        optimizer is not None
        and
        optimizer_state is not None
    ):

        optimizer.load_state_dict(
            optimizer_state
        )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )


    if (
        scheduler is not None
        and
        scheduler_state is not None
    ):

        scheduler.load_state_dict(
            scheduler_state
        )


    # ========================================================
    # RETURN TRAINING STATE
    # ========================================================

    return {
        "step":
            int(
                checkpoint.get(
                    "step",
                    0,
                )
            ),

        "epoch":
            int(
                checkpoint.get(
                    "epoch",
                    0,
                )
            ),

        "loss":
            checkpoint.get(
                "loss"
            ),

        "metadata":
            checkpoint.get(
                "metadata",
                {},
            ),
    }


# ============================================================
# READ CHECKPOINT METADATA ONLY
# ============================================================

def read_checkpoint_metadata(
    path,
    map_location="cpu",
):
    """
    Read metadata without loading it into a model.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {path}"
        )


    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )


    return {
        "checkpoint_format":
            checkpoint.get(
                "checkpoint_format"
            ),

        "step":
            checkpoint.get(
                "step",
                0,
            ),

        "epoch":
            checkpoint.get(
                "epoch",
                0,
            ),

        "loss":
            checkpoint.get(
                "loss"
            ),

        "metadata":
            checkpoint.get(
                "metadata",
                {},
            ),
    }


# ============================================================
# EXPORT METADATA TO JSON
# ============================================================

def export_metadata(
    checkpoint_path,
    json_path,
):
    """
    Export checkpoint metadata into JSON.
    """

    data = read_checkpoint_metadata(
        checkpoint_path
    )


    directory = os.path.dirname(
        os.path.abspath(json_path)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )


    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


    return json_path


# ============================================================
# CHECKPOINT SUMMARY
# ============================================================

def print_checkpoint_summary(
    path
):
    """
    Print human-readable checkpoint information.
    """

    data = read_checkpoint_metadata(
        path
    )


    metadata = data.get(
        "metadata",
        {}
    )


    print("=" * 60)
    print("VENLA CHECKPOINT")
    print("=" * 60)

    print()

    print(
        "File:",
        path,
    )

    print(
        "Format:",
        data.get(
            "checkpoint_format"
        ),
    )

    print(
        "Step:",
        data.get(
            "step"
        ),
    )

    print(
        "Epoch:",
        data.get(
            "epoch"
        ),
    )

    print(
        "Loss:",
        data.get(
            "loss"
        ),
    )

    print()

    print(
        "Model:",
        metadata.get(
            "model_name"
        ),
    )

    print(
        "Version:",
        metadata.get(
            "model_version"
        ),
    )

    print(
        "Parameters:",
        metadata.get(
            "parameters"
        ),
    )

    print(
        "Trainable:",
        metadata.get(
            "trainable_parameters"
        ),
    )

    print()

    print("=" * 60)


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(
    path
):
    """
    Validate checkpoint structure.
    """

    if not os.path.exists(path):

        return {
            "valid": False,
            "reason":
                "file_not_found",
        }


    try:

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


        required_keys = [
            "checkpoint_format",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "step",
            "epoch",
            "loss",
            "metadata",
        ]


        missing = [
            key
            for key in required_keys
            if key not in checkpoint
        ]


        if missing:

            return {
                "valid": False,
                "reason":
                    "missing_keys",
                "missing":
                    missing,
            }


        if (
            checkpoint[
                "checkpoint_format"
            ]
            != CHECKPOINT_FORMAT
        ):

            return {
                "valid": False,
                "reason":
                    "invalid_format",
            }


        return {
            "valid": True,
            "reason": "ok",
            "step":
                checkpoint.get(
                    "step",
                    0,
                ),
            "epoch":
                checkpoint.get(
                    "epoch",
                    0,
                ),
            "loss":
                checkpoint.get(
                    "loss"
                ),
        }


    except Exception as error:

        return {
            "valid": False,
            "reason":
                "load_error",
            "error":
                str(error),
        }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VENLA CHECKPOINT MANAGER V1")
    print("=" * 60)

    print()

    print(
        "Format:",
        CHECKPOINT_FORMAT
    )

    print()

    print(
        "Checkpoint manager siap digunakan."
    )

    print()

    print("=" * 60)
