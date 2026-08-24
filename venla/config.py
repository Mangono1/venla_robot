"""
VENLA V0.1
Configuration
"""

# ============================================================
# MODEL IDENTITY
# ============================================================

MODEL_NAME = "VENLA"
MODEL_VERSION = "V0.1"


# ============================================================
# TOKENIZER
# ============================================================

VOCAB_SIZE = 32768


# ============================================================
# CONTEXT
# ============================================================

CONTEXT_LENGTH = 512


# ============================================================
# TRANSFORMER
# ============================================================

EMBED_DIM = 768

NUM_LAYERS = 9

NUM_HEADS = 12

MLP_DIM = 3584


# ============================================================
# TRAINING DEFAULTS
# ============================================================

DEFAULT_BATCH_SIZE = 2

DEFAULT_LEARNING_RATE = 3e-4

DEFAULT_WEIGHT_DECAY = 0.1

DEFAULT_GRADIENT_CLIP = 1.0

DEFAULT_WARMUP_STEPS = 500

DEFAULT_MAX_STEPS = 10000


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_VERSION = "V0.1"


# ============================================================
# DATASET
# ============================================================

DEFAULT_SEQUENCE_LENGTH = CONTEXT_LENGTH


# ============================================================
# DEVICE
# ============================================================

USE_CUDA = True


# ============================================================
# CONFIG DICTIONARY
# ============================================================

MODEL_CONFIG = {
    "model_name": MODEL_NAME,
    "version": MODEL_VERSION,

    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,

    "embed_dim": EMBED_DIM,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "mlp_dim": MLP_DIM,
}


TRAINING_CONFIG = {
    "batch_size": DEFAULT_BATCH_SIZE,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
    "gradient_clip": DEFAULT_GRADIENT_CLIP,
    "warmup_steps": DEFAULT_WARMUP_STEPS,
    "max_steps": DEFAULT_MAX_STEPS,
}


def get_model_config():
    """
    Return a copy of the model configuration.
    """

    return dict(MODEL_CONFIG)


def get_training_config():
    """
    Return a copy of the training configuration.
    """

    return dict(TRAINING_CONFIG)


def print_config():
    """
    Print VENLA configuration.
    """

    print("=" * 60)
    print("VENLA V0.1 CONFIGURATION")
    print("=" * 60)

    print()

    print(
        "Model       :",
        MODEL_NAME
    )

    print(
        "Version     :",
        MODEL_VERSION
    )

    print()

    print(
        "Vocabulary  :",
        VOCAB_SIZE
    )

    print(
        "Context     :",
        CONTEXT_LENGTH
    )

    print(
        "Embedding   :",
        EMBED_DIM
    )

    print(
        "Layers      :",
        NUM_LAYERS
    )

    print(
        "Heads       :",
        NUM_HEADS
    )

    print(
        "MLP         :",
        MLP_DIM
    )

    print()

    print(
        "Batch size  :",
        DEFAULT_BATCH_SIZE
    )

    print(
        "LR          :",
        DEFAULT_LEARNING_RATE
    )

    print(
        "Weight decay:",
        DEFAULT_WEIGHT_DECAY
    )

    print(
        "Max steps   :",
        DEFAULT_MAX_STEPS
    )

    print()

    print("=" * 60)
