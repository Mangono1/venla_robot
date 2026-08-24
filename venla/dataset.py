"""
VENLA V0.1
Dataset Engine V1

Mengubah teks menjadi dataset language modeling:

Input:
    token[0 : context_length]

Target:
    token[1 : context_length + 1]

Contoh:

    teks: ABCDEFG

    input : ABCDE
    target: BCDEF
"""

import os
import json

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_VOCAB_SIZE = 32768


# ============================================================
# VENLA TEXT DATASET
# ============================================================

class VENLATextDataset(Dataset):
    """
    Dataset language modeling berbasis token.

    Dataset dibuat dari file teks.
    """

    def __init__(
        self,
        tokens,
        context_length=DEFAULT_CONTEXT_LENGTH,
    ):

        self.context_length = int(
            context_length
        )

        self.tokens = torch.tensor(
            tokens,
            dtype=torch.long,
        )

        if len(self.tokens) <= self.context_length:

            raise ValueError(
                "Jumlah token harus lebih besar "
                "daripada context_length."
            )

        self.sequence_count = (
            len(self.tokens)
            - self.context_length
        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return self.sequence_count


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        start = int(index)

        end = (
            start
            + self.context_length
            + 1
        )

        sequence = self.tokens[
            start:end
        ]

        input_ids = sequence[
            :-1
        ]

        targets = sequence[
            1:
        ]

        return (
            input_ids,
            targets,
        )


# ============================================================
# DATASET ENGINE
# ============================================================

class VENLADatasetEngine:
    """
    Dataset Engine V1.

    Bertanggung jawab untuk:

    text
        ↓
    tokenizer
        ↓
    tokens
        ↓
    dataset
        ↓
    dataloader
    """

    def __init__(
        self,
        tokenizer,
        context_length=DEFAULT_CONTEXT_LENGTH,
    ):

        self.tokenizer = tokenizer

        self.context_length = int(
            context_length
        )

        self.tokens = None

        self.dataset = None

        self.dataloader = None


    # ========================================================
    # LOAD TEXT
    # ========================================================

    def load_text(
        self,
        path,
    ):

        if not os.path.exists(path):

            raise FileNotFoundError(
                "Dataset tidak ditemukan: "
                + path
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            text = file.read()

        return text


    # ========================================================
    # TOKENIZE TEXT
    # ========================================================

    def tokenize_text(
        self,
        text,
    ):

        tokens = self.tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        self.tokens = tokens

        return tokens


    # ========================================================
    # BUILD DATASET
    # ========================================================

    def build_dataset(
        self,
        tokens=None,
    ):

        if tokens is not None:

            self.tokens = tokens

        if self.tokens is None:

            raise RuntimeError(
                "Token belum tersedia."
            )

        self.dataset = VENLATextDataset(
            tokens=self.tokens,
            context_length=self.context_length,
        )

        return self.dataset


    # ========================================================
    # BUILD DATALOADER
    # ========================================================

    def build_dataloader(
        self,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    ):

        if self.dataset is None:

            raise RuntimeError(
                "Dataset belum dibuat."
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

        return self.dataloader


    # ========================================================
    # BUILD FROM FILE
    # ========================================================

    def build_from_file(
        self,
        path,
    ):

        text = self.load_text(
            path
        )

        tokens = self.tokenize_text(
            text
        )

        dataset = self.build_dataset(
            tokens
        )

        return dataset


    # ========================================================
    # SAMPLE
    # ========================================================

    def get_sample(
        self,
        index=0,
    ):

        if self.dataset is None:

            raise RuntimeError(
                "Dataset belum dibuat."
            )

        return self.dataset[
            index
        ]


    # ========================================================
    # INFO
    # ========================================================

    def info(self):

        token_count = (
            0
            if self.tokens is None
            else len(self.tokens)
        )

        dataset_sequences = (
            0
            if self.dataset is None
            else len(self.dataset)
        )

        return {
            "context_length":
                self.context_length,

            "token_count":
                token_count,

            "dataset_sequences":
                dataset_sequences,
        }


# ============================================================
# SAVE TOKENIZED DATASET
# ============================================================

def save_tokens(
    tokens,
    path,
):

    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    data = {
        "format":
            "VENLA_TOKEN_DATA_V1",

        "token_count":
            len(tokens),

        "tokens":
            list(
                map(
                    int,
                    tokens,
                )
            ),
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
        )

    return path


# ============================================================
# LOAD TOKENIZED DATASET
# ============================================================

def load_tokens(
    path,
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            "Token dataset tidak ditemukan: "
            + path
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if data.get(
        "format"
    ) != "VENLA_TOKEN_DATA_V1":

        raise RuntimeError(
            "Format token dataset tidak dikenal."
        )

    tokens = data[
        "tokens"
    ]

    return list(
        map(
            int,
            tokens,
        )
    )


# ============================================================
# DATASET TEST
# ============================================================

def test_dataset(
    tokenizer,
):

    print("=" * 60)
    print("VENLA V0.1 - DATASET ENGINE V1")
    print("=" * 60)

    print()

    # --------------------------------------------------------
    # TEST TEXT
    # --------------------------------------------------------

    text = """
VENLA adalah model bahasa yang dibuat dari nol.

Model ini menggunakan arsitektur Transformer
decoder-only.

Tujuan proyek ini adalah membangun model bahasa
sendiri, mulai dari tokenizer, dataset, training,
checkpoint, evaluasi, sampai inference.

Indonesia adalah bahasa utama dalam dataset awal.

VENLA dikembangkan secara modular agar setiap
komponen dapat diuji secara terpisah sebelum
digabungkan menjadi sistem training lengkap.
"""

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    engine = VENLADatasetEngine(
        tokenizer=tokenizer,
        context_length=512,
    )

    tokens = engine.tokenize_text(
        text
    )

    print(
        "Characters:",
        len(text)
    )

    print(
        "Tokens:",
        len(tokens)
    )

    print()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = engine.build_dataset(
        tokens
    )

    print(
        "Dataset sequences:",
        len(dataset)
    )

    print()

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    input_ids, targets = (
        engine.get_sample(0)
    )

    print(
        "Sample:"
    )

    print(
        "Input shape:",
        tuple(
            input_ids.shape
        )
    )

    print(
        "Target shape:",
        tuple(
            targets.shape
        )
    )

    print()

    decoded_input = tokenizer.decode(
        input_ids.tolist()
    )

    print(
        "Decoded input:"
    )

    print(
        decoded_input[:500]
    )

    print()

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    dataloader = engine.build_dataloader(
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    batch_input, batch_target = next(
        iter(dataloader)
    )

    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print()

    print(
        "Batch input:",
        tuple(
            batch_input.shape
        )
    )

    print(
        "Batch target:",
        tuple(
            batch_target.shape
        )
    )

    print()

    assert batch_input.shape == (
        2,
        512,
    )

    assert batch_target.shape == (
        2,
        512,
    )

    # --------------------------------------------------------
    # TOKEN SAVE / LOAD
    # --------------------------------------------------------

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:

        token_path = os.path.join(
            temp_dir,
            "venla_tokens.json",
        )

        save_tokens(
            tokens,
            token_path,
        )

        loaded_tokens = load_tokens(
            token_path
        )

        assert tokens == loaded_tokens

    print(
        "✅ TOKEN DATA SAVE/LOAD BERHASIL"
    )

    print()

    print(
        "Input tokens:",
        input_ids[:30].tolist()
    )

    print()

    print("=" * 60)
    print("✅ DATASET ENGINE V1 BERHASIL")
    print("=" * 60)

    return engine


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import sys

    ROOT_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    if ROOT_DIR not in sys.path:

        sys.path.insert(
            0,
            ROOT_DIR,
        )

    from venla.tokenizer import (
        VENLATokenizer
    )

    tokenizer = VENLATokenizer(
        vocab_size=32768
    )

    test_dataset(
        tokenizer
    )
