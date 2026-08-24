"""
VENLA V0.1
Training Orchestrator

Alur:

    Config
      ↓
    Tokenizer
      ↓
    Dataset
      ↓
    Model
      ↓
    Trainer
      ↓
    Checkpoint
      ↓
    Evaluator
      ↓
    Supabase Sync

Script ini menjadi entry point utama
untuk training VENLA.
"""

import os
import sys
import json
import time
import argparse

import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ============================================================
# VENLA IMPORTS
# ============================================================

from venla.model import VENLA

from venla.tokenizer import (
    VENLATokenizer,
)

from venla.dataset import (
    VENLADatasetEngine,
)

from venla.trainer import (
    VENLATrainer,
)

from venla.evaluator import (
    VENLAEvaluator,
)

from venla.supabase import (
    VENLASupabase,
    SupabaseError,
)


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_DATASET = os.path.join(
    PROJECT_ROOT,
    "data",
    "train.txt",
)

DEFAULT_TOKENIZER = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "tokenizer",
    "venla_tokenizer_v1.json",
)

DEFAULT_CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "checkpoints",
)

DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "configs",
    "training_config.json",
)

DEFAULT_LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "logs",
)


# ============================================================
# DEFAULT TRAINING CONFIG
# ============================================================

DEFAULT_CONFIG_DATA = {

    "model": {

        "vocab_size":
            32768,

        "context_length":
            512,

        "embed_dim":
            768,

        "num_layers":
            9,

        "num_heads":
            12,

        "mlp_dim":
            3584,
    },

    "training": {

        "batch_size":
            2,

        "learning_rate":
            0.0003,

        "weight_decay":
            0.1,

        "gradient_clip":
            1.0,

        "max_steps":
            1000,

        "log_interval":
            10,

        "checkpoint_interval":
            100,

        "use_amp":
            True,
    },

    "dataset": {

        "num_workers":
            0,

        "shuffle":
            True,

        "drop_last":
            False,

        "pin_memory":
            True,
    },

    "supabase": {

        "bucket":
            "venla",

        "enabled":
            True,

        "upload_checkpoints":
            True,

        "upload_logs":
            True,
    },
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

def ensure_directories():

    directories = [

        os.path.dirname(
            DEFAULT_TOKENIZER
        ),

        DEFAULT_CHECKPOINT_DIR,

        os.path.dirname(
            DEFAULT_CONFIG
        ),

        DEFAULT_LOG_DIR,

        os.path.join(
            PROJECT_ROOT,
            "data",
        ),
    ]

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# CONFIG SAVE
# ============================================================

def save_config(
    config,
    path=DEFAULT_CONFIG,
):

    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path


# ============================================================
# CONFIG LOAD
# ============================================================

def load_config(
    path=DEFAULT_CONFIG,
):

    if not os.path.exists(path):

        save_config(
            DEFAULT_CONFIG_DATA,
            path,
        )

        return DEFAULT_CONFIG_DATA


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# DEVICE INFORMATION
# ============================================================

def print_device():

    print("=" * 60)
    print("VENLA V0.1 - HARDWARE")
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


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        props = (
            torch.cuda.get_device_properties(
                0
            )
        )

        vram = (
            props.total_memory
            / (
                1024 ** 3
            )
        )

        print(
            "VRAM:",
            f"{vram:.2f} GB"
        )


    else:

        print(
            "GPU: CPU"
        )


    print()

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# TOKENIZER
# ============================================================

def prepare_tokenizer(
    vocab_size,
    tokenizer_path=DEFAULT_TOKENIZER,
):

    print("=" * 60)
    print("TOKENIZER")
    print("=" * 60)

    print()

    if os.path.exists(
        tokenizer_path
    ):

        print(
            "Loading tokenizer:"
        )

        print(
            tokenizer_path
        )

        tokenizer = (
            VENLATokenizer.load(
                tokenizer_path
            )
        )

    else:

        print(
            "Creating tokenizer..."
        )

        tokenizer = VENLATokenizer(
            vocab_size=vocab_size
        )

        tokenizer.save(
            tokenizer_path
        )

        print(
            "Tokenizer saved:"
        )

        print(
            tokenizer_path
        )


    tokenizer.validate()

    print()

    print(
        "Vocabulary:",
        tokenizer.get_vocab_size()
    )

    print()

    return tokenizer


# ============================================================
# DATASET
# ============================================================

def prepare_dataset(
    tokenizer,
    dataset_path,
    context_length,
    batch_size,
    num_workers,
    shuffle,
    drop_last,
    pin_memory,
):

    print("=" * 60)
    print("DATASET")
    print("=" * 60)

    print()

    if not os.path.exists(
        dataset_path
    ):

        raise FileNotFoundError(
            "Dataset tidak ditemukan:\n"
            + dataset_path
            + "\n\n"
            "Buat file data/train.txt terlebih dahulu."
        )


    print(
        "Dataset:",
        dataset_path
    )


    engine = VENLADatasetEngine(
        tokenizer=tokenizer,
        context_length=context_length,
    )


    text = engine.load_text(
        dataset_path
    )


    print(
        "Characters:",
        len(text)
    )


    tokens = engine.tokenize_text(
        text
    )


    print(
        "Tokens:",
        len(tokens)
    )


    dataset = engine.build_dataset(
        tokens
    )


    print(
        "Sequences:",
        len(dataset)
    )


    dataloader = engine.build_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


    print()

    print(
        "Batch size:",
        batch_size
    )

    print(
        "Context:",
        context_length
    )

    print()

    return (
        engine,
        dataset,
        dataloader,
    )


# ============================================================
# MODEL
# ============================================================

def prepare_model(
    model_config,
    device,
):

    print("=" * 60)
    print("MODEL")
    print("=" * 60)

    print()

    model = VENLA(
        vocab_size=model_config[
            "vocab_size"
        ],

        context_length=model_config[
            "context_length"
        ],

        embed_dim=model_config[
            "embed_dim"
        ],

        num_layers=model_config[
            "num_layers"
        ],

        num_heads=model_config[
            "num_heads"
        ],

        mlp_dim=model_config[
            "mlp_dim"
        ],
    )


    model = model.to(
        device
    )


    parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )


    print(
        "Vocabulary:",
        model_config[
            "vocab_size"
        ]
    )

    print(
        "Context:",
        model_config[
            "context_length"
        ]
    )

    print(
        "Embedding:",
        model_config[
            "embed_dim"
        ]
    )

    print(
        "Layers:",
        model_config[
            "num_layers"
        ]
    )

    print(
        "Heads:",
        model_config[
            "num_heads"
        ]
    )

    print(
        "MLP:",
        model_config[
            "mlp_dim"
        ]
    )

    print()

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    print(
        "Trainable:",
        f"{trainable:,}"
    )

    print()

    return model


# ============================================================
# SUPABASE
# ============================================================

def prepare_supabase(
    config,
):

    supabase_config = config[
        "supabase"
    ]

    if not supabase_config.get(
        "enabled",
        True,
    ):

        print(
            "Supabase disabled."
        )

        return None


    url = os.environ.get(
        "SUPABASE_URL"
    )

    key = os.environ.get(
        "SUPABASE_KEY"
    )


    if not url or not key:

        print(
            "⚠️ Supabase credential belum tersedia."
        )

        print(
            "Training tetap dapat berjalan."
        )

        return None


    try:

        client = VENLASupabase(
            url=url,
            key=key,
            bucket=supabase_config.get(
                "bucket",
                "venla",
            ),
        )


        print(
            "Supabase:",
            client.url
        )

        print(
            "Bucket:",
            client.bucket
        )


        return client


    except Exception as error:

        print(
            "⚠️ Supabase initialization gagal:"
        )

        print(
            str(error)
        )

        return None


# ============================================================
# SUPABASE UPLOAD
# ============================================================

def upload_to_supabase(
    client,
    local_path,
    remote_path,
):

    if client is None:

        return False


    if not os.path.exists(
        local_path
    ):

        print(
            "Upload dilewati. File tidak ditemukan:"
        )

        print(
            local_path
        )

        return False


    try:

        result = client.upload_file(
            local_path=local_path,
            remote_path=remote_path,
            overwrite=True,
        )


        print(
            "☁️ Supabase upload:"
        )

        print(
            remote_path
        )

        print(
            "Size:",
            result["size"],
            "bytes"
        )


        return True


    except Exception as error:

        print(
            "⚠️ Supabase upload gagal:"
        )

        print(
            str(error)
        )

        return False


# ============================================================
# TRAINING LOG
# ============================================================

def append_training_log(
    result,
    path=None,
):

    if path is None:

        path = os.path.join(
            DEFAULT_LOG_DIR,
            "training.jsonl",
        )


    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )


    record = {
        "timestamp":
            time.time(),

        "result":
            result,
    }


    with open(
        path,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


    return path


# ============================================================
# MAIN TRAINING
# ============================================================

def run_training(
    dataset_path=DEFAULT_DATASET,
    max_steps=None,
    resume=None,
):

    ensure_directories()


    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = load_config()

    save_config(
        config
    )


    model_config = config[
        "model"
    ]

    training_config = config[
        "training"
    ]

    dataset_config = config[
        "dataset"
    ]


    if max_steps is None:

        max_steps = training_config[
            "max_steps"
        ]


    # --------------------------------------------------------
    # HARDWARE
    # --------------------------------------------------------

    device = print_device()


    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = prepare_tokenizer(
        vocab_size=model_config[
            "vocab_size"
        ],
        tokenizer_path=DEFAULT_TOKENIZER,
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    (
        engine,
        dataset,
        dataloader,
    ) = prepare_dataset(

        tokenizer=tokenizer,

        dataset_path=dataset_path,

        context_length=model_config[
            "context_length"
        ],

        batch_size=training_config[
            "batch_size"
        ],

        num_workers=dataset_config[
            "num_workers"
        ],

        shuffle=dataset_config[
            "shuffle"
        ],

        drop_last=dataset_config[
            "drop_last"
        ],

        pin_memory=dataset_config[
            "pin_memory"
        ],
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = prepare_model(
        model_config,
        device,
    )


    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = VENLATrainer(

        model=model,

        dataloader=dataloader,

        device=device,

        learning_rate=training_config[
            "learning_rate"
        ],

        weight_decay=training_config[
            "weight_decay"
        ],

        gradient_clip=training_config[
            "gradient_clip"
        ],

        max_steps=max_steps,

        log_interval=training_config[
            "log_interval"
        ],

        checkpoint_interval=training_config[
            "checkpoint_interval"
        ],

        checkpoint_dir=(
            DEFAULT_CHECKPOINT_DIR
        ),

        use_amp=training_config[
            "use_amp"
        ],
    )


    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if resume:

        print("=" * 60)
        print("RESUME TRAINING")
        print("=" * 60)

        print()

        print(
            "Checkpoint:",
            resume
        )

        trainer.load_checkpoint(
            resume
        )

        print(
            "Resume step:",
            trainer.step
        )

        print()


    # --------------------------------------------------------
    # SUPABASE
    # --------------------------------------------------------

    supabase = prepare_supabase(
        config
    )


    # --------------------------------------------------------
    # SAVE CONFIG
    # --------------------------------------------------------

    config_remote = (
        "configs/"
        "training_config.json"
    )


    upload_to_supabase(
        supabase,
        DEFAULT_CONFIG,
        config_remote,
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    result = trainer.train(
        max_steps=max_steps
    )


    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    log_path = append_training_log(
        result
    )


    # --------------------------------------------------------
    # UPLOAD FINAL CHECKPOINT
    # --------------------------------------------------------

    if config[
        "supabase"
    ].get(
        "upload_checkpoints",
        True,
    ):

        upload_to_supabase(
            supabase,

            result[
                "checkpoint"
            ],

            "checkpoints/final.pt",
        )


        latest_path = os.path.join(
            DEFAULT_CHECKPOINT_DIR,
            "latest.pt",
        )


        upload_to_supabase(
            supabase,
            latest_path,
            "checkpoints/latest.pt",
        )


    # --------------------------------------------------------
    # UPLOAD LOG
    # --------------------------------------------------------

    if config[
        "supabase"
    ].get(
        "upload_logs",
        True,
    ):

        upload_to_supabase(
            supabase,

            log_path,

            "logs/training.jsonl",
        )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    print()


    evaluator = VENLAEvaluator(
        model=model,
        device=device,
    )


    evaluation = evaluator.evaluate(
        dataloader,
        max_batches=10,
    )


    evaluator.print_result(
        evaluation
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    final_result = {

        "training":
            result,

        "evaluation":
            evaluation,

        "device":
            str(device),

        "model":
            model_config,

        "dataset":
            engine.info(),
    }


    result_path = os.path.join(
        DEFAULT_LOG_DIR,
        "final_result.json",
    )


    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_result,
            file,
            indent=2,
            ensure_ascii=False,
        )


    upload_to_supabase(
        supabase,
        result_path,
        "logs/final_result.json",
    )


    print()

    print("=" * 60)
    print("VENLA TRAINING PIPELINE SELESAI")
    print("=" * 60)

    print()

    print(
        "Final loss:",
        result[
            "loss"
        ]
    )

    print(
        "Perplexity:",
        evaluation[
            "perplexity"
        ]
    )

    print(
        "Checkpoint:",
        result[
            "checkpoint"
        ]
    )

    print()

    return final_result


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "VENLA V0.1 Training"
        )
    )


    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=(
            "Path dataset training"
        ),
    )


    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Jumlah training steps"
        ),
    )


    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path checkpoint "
            "untuk resume training"
        ),
    )


    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    args = parse_arguments()


    run_training(

        dataset_path=args.dataset,

        max_steps=args.steps,

        resume=args.resume,
    )


if __name__ == "__main__":

    main()
