"""
VENLA V0.1
Model Test Suite

Mengetes:
- konfigurasi
- model creation
- parameter count
- forward pass
- loss
- backward pass
- checkpoint save/load
"""

import os
import sys
import tempfile

import torch


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if ROOT_DIR not in sys.path:
    sys.path.insert(
        0,
        ROOT_DIR,
    )


# ============================================================
# VENLA IMPORTS
# ============================================================

from venla.model import VENLA
from venla.config import (
    MODEL_NAME,
    MODEL_VERSION,
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    EMBED_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    MLP_DIM,
)
from venla.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    validate_checkpoint,
)


# ============================================================
# EXPECTED PARAMETERS
# ============================================================

EXPECTED_PARAMETERS = 96_433_920


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

def print_header():

    print("=" * 60)
    print("VENLA V0.1 - COMPLETE MODEL TEST")
    print("=" * 60)

    print()

    print(
        "Model:",
        MODEL_NAME
    )

    print(
        "Version:",
        MODEL_VERSION
    )

    print()


# ============================================================
# HARDWARE TEST
# ============================================================

def test_hardware(device):

    print("=" * 60)
    print("HARDWARE")
    print("=" * 60)

    print()

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.cuda.is_available()
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        props = torch.cuda.get_device_properties(0)

        vram_gb = (
            props.total_memory
            / 1024**3
        )

        print(
            "VRAM:",
            f"{vram_gb:.2f} GB"
        )

    print()

    print("✅ HARDWARE TEST OK")

    print()


# ============================================================
# CONFIG TEST
# ============================================================

def test_config():

    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)

    print()

    print(
        "Vocabulary:",
        VOCAB_SIZE
    )

    print(
        "Context:",
        CONTEXT_LENGTH
    )

    print(
        "Embedding:",
        EMBED_DIM
    )

    print(
        "Layers:",
        NUM_LAYERS
    )

    print(
        "Heads:",
        NUM_HEADS
    )

    print(
        "MLP:",
        MLP_DIM
    )

    print()

    assert VOCAB_SIZE == 32768

    assert CONTEXT_LENGTH == 512

    assert EMBED_DIM == 768

    assert NUM_LAYERS == 9

    assert NUM_HEADS == 12

    assert MLP_DIM == 3584

    print(
        "✅ CONFIGURATION TEST OK"
    )

    print()


# ============================================================
# MODEL CREATION
# ============================================================

def test_model_creation(device):

    print("=" * 60)
    print("MODEL CREATION")
    print("=" * 60)

    print()

    model = VENLA().to(device)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    print(
        "Trainable:",
        f"{trainable:,}"
    )

    print()

    assert parameters == EXPECTED_PARAMETERS

    assert trainable == EXPECTED_PARAMETERS

    print(
        "✅ PARAMETER COUNT COCOK"
    )

    print()

    return model


# ============================================================
# FORWARD TEST
# ============================================================

def test_forward(
    model,
    device,
):

    print("=" * 60)
    print("FORWARD TEST")
    print("=" * 60)

    print()

    batch_size = 1

    sequence_length = 64

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (
            batch_size,
            sequence_length,
        ),
        device=device,
    )

    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (
            batch_size,
            sequence_length,
        ),
        device=device,
    )

    model.eval()

    with torch.no_grad():

        logits, loss = model(
            input_ids,
            targets,
        )

    print(
        "Input shape:",
        tuple(
            input_ids.shape
        )
    )

    print(
        "Logits shape:",
        tuple(
            logits.shape
        )
    )

    print(
        "Loss:",
        float(loss)
    )

    print()

    assert logits.shape == (
        batch_size,
        sequence_length,
        VOCAB_SIZE,
    )

    assert loss.ndim == 0

    assert torch.isfinite(loss)

    print(
        "✅ FORWARD TEST OK"
    )

    print()

    return float(loss)


# ============================================================
# BACKWARD TEST
# ============================================================

def test_backward(
    model,
    device,
):

    print("=" * 60)
    print("BACKWARD TEST")
    print("=" * 60)

    print()

    batch_size = 1

    sequence_length = 32

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (
            batch_size,
            sequence_length,
        ),
        device=device,
    )

    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (
            batch_size,
            sequence_length,
        ),
        device=device,
    )

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    logits, loss = model(
        input_ids,
        targets,
    )

    loss.backward()

    gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_count += 1

    print(
        "Loss:",
        float(loss)
    )

    print(
        "Parameters with gradient:",
        gradient_count
    )

    print()

    assert gradient_count > 0

    print(
        "✅ BACKWARD TEST OK"
    )

    print()


# ============================================================
# CHECKPOINT TEST
# ============================================================

def test_checkpoint(
    model,
    device,
):

    print("=" * 60)
    print("CHECKPOINT TEST")
    print("=" * 60)

    print()

    with tempfile.TemporaryDirectory() as temp_dir:

        checkpoint_path = os.path.join(
            temp_dir,
            "venla_test.pt",
        )

        print(
            "Saving checkpoint..."
        )

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            step=123,
            epoch=4,
            loss=1.2345,
        )

        assert os.path.exists(
            checkpoint_path
        )

        file_size = (
            os.path.getsize(
                checkpoint_path
            )
            / 1024**2
        )

        print(
            "Checkpoint size:",
            f"{file_size:.2f} MB"
        )

        print()

        validation = validate_checkpoint(
            checkpoint_path
        )

        print(
            "Validation:",
            validation
        )

        assert validation["valid"]

        print()

        # ----------------------------------------------------
        # NEW MODEL
        # ----------------------------------------------------

        new_model = VENLA().to(device)

        state = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            map_location=device,
        )

        print(
            "Loaded step:",
            state["step"]
        )

        print(
            "Loaded epoch:",
            state["epoch"]
        )

        print(
            "Loaded loss:",
            state["loss"]
        )

        assert state["step"] == 123

        assert state["epoch"] == 4

        assert state["loss"] == 1.2345

        # ----------------------------------------------------
        # PARAMETER COMPARISON
        # ----------------------------------------------------

        original_parameters = list(
            model.parameters()
        )

        loaded_parameters = list(
            new_model.parameters()
        )

        assert len(
            original_parameters
        ) == len(
            loaded_parameters
        )

        for original, loaded in zip(
            original_parameters,
            loaded_parameters,
        ):

            assert torch.equal(
                original.detach().cpu(),
                loaded.detach().cpu(),
            )

    print()

    print(
        "✅ CHECKPOINT TEST OK"
    )

    print()


# ============================================================
# GPU MEMORY
# ============================================================

def print_gpu_memory():

    if not torch.cuda.is_available():

        return

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print("=" * 60)
    print("GPU MEMORY")
    print("=" * 60)

    print()

    print(
        "Allocated:",
        f"{allocated:.3f} GB"
    )

    print(
        "Reserved :",
        f"{reserved:.3f} GB"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    device = get_device()

    test_hardware(
        device
    )

    test_config()

    model = test_model_creation(
        device
    )

    test_forward(
        model,
        device,
    )

    test_backward(
        model,
        device,
    )

    test_checkpoint(
        model,
        device,
    )

    print_gpu_memory()

    print("=" * 60)
    print("🎉 VENLA V0.1 - ALL TESTS PASSED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
