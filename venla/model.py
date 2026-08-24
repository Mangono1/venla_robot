# ============================================================
# VENLA V0.1
# MODEL SOURCE GENERATOR
# ============================================================

import os
import textwrap

REPO_DIR = "/content/venla_robot"
VENLA_DIR = os.path.join(REPO_DIR, "venla")

os.makedirs(VENLA_DIR, exist_ok=True)


MODEL_CODE = r'''
"""
VENLA V0.1
Transformer Decoder-Only Language Model

Architecture:
    Vocabulary : 32,768
    Context    : 512
    Embedding  : 768
    Layers     : 9
    Heads      : 12
    MLP        : 3,584

Target:
    ~100M parameters
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MODEL CONFIGURATION
# ============================================================

VOCAB_SIZE = 32768
CONTEXT_LENGTH = 512

EMBED_DIM = 768
NUM_LAYERS = 9
NUM_HEADS = 12
MLP_DIM = 3584

MODEL_NAME = "VENLA"
MODEL_VERSION = "V0.1"


# ============================================================
# CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Tokens can only attend to:
        current token
        previous tokens

    Future tokens are masked.
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        context_length=CONTEXT_LENGTH,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim harus habis dibagi num_heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = (
            embed_dim // num_heads
        )

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3,
        )

        self.proj = nn.Linear(
            embed_dim,
            embed_dim,
        )

        causal_mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x):
        """
        x:
            [batch, sequence, embedding]
        """

        batch_size, sequence_length, embed_dim = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        output = attention_weights @ v

        output = output.transpose(
            1,
            2,
        ).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embed_dim,
        )

        return self.proj(output)


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class MLP(nn.Module):
    """
    Transformer feed-forward network.
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        mlp_dim=MLP_DIM,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            embed_dim,
            mlp_dim,
        )

        self.fc2 = nn.Linear(
            mlp_dim,
            embed_dim,
        )

    def forward(self, x):

        x = self.fc1(x)

        x = F.gelu(x)

        x = self.fc2(x)

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer decoder block.
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        context_length=CONTEXT_LENGTH,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# VENLA MODEL
# ============================================================

class VENLA(nn.Module):
    """
    VENLA V0.1 decoder-only language model.

    Weight tying:
        token embedding weights are reused
        by the language-model output head.
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        # ----------------------------------------------------
        # TOKEN EMBEDDING
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        # ----------------------------------------------------
        # POSITION EMBEDDING
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            context_length,
            embed_dim,
        )

        # ----------------------------------------------------
        # TRANSFORMER BLOCKS
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    context_length=context_length,
                )
                for _ in range(num_layers)
            ]
        )

        # ----------------------------------------------------
        # FINAL NORMALIZATION
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(
            embed_dim
        )

        # ----------------------------------------------------
        # WEIGHT INITIALIZATION
        # ----------------------------------------------------

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.LayerNorm,
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        input_ids,
        targets=None,
    ):
        """
        Parameters
        ----------
        input_ids:
            Tensor [batch, sequence]

        targets:
            Tensor [batch, sequence]

        Returns
        -------
        logits:
            Tensor [batch, sequence, vocab]

        loss:
            Cross entropy loss or None
        """

        batch_size, sequence_length = (
            input_ids.shape
        )

        if sequence_length > self.context_length:

            raise ValueError(
                f"Sequence length "
                f"{sequence_length} melebihi "
                f"context length "
                f"{self.context_length}."
            )

        # ----------------------------------------------------
        # POSITION IDS
        # ----------------------------------------------------

        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        x = (
            self.token_embedding(input_ids)
            +
            self.position_embedding(position_ids)
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # FINAL NORMALIZATION
        # ----------------------------------------------------

        x = self.norm(x)

        # ----------------------------------------------------
        # LANGUAGE MODEL HEAD
        #
        # Weight tying:
        # output projection uses token embedding matrix.
        # ----------------------------------------------------

        logits = F.linear(
            x,
            self.token_embedding.weight,
        )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.vocab_size,
                ),
                targets.reshape(
                    -1,
                ),
            )

        return logits, loss

    def parameter_count(self):

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    def trainable_parameter_count(self):

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def model_config(self):

        return {
            "model_name": MODEL_NAME,
            "version": MODEL_VERSION,
            "architecture": "Transformer decoder-only",
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "parameters": self.parameter_count(),
            "trainable_parameters":
                self.trainable_parameter_count(),
            "weight_tying": True,
        }


# ============================================================
# FACTORY
# ============================================================

def create_model(
    device=None,
):
    """
    Create VENLA V0.1.
    """

    model = VENLA()

    if device is not None:

        model = model.to(device)

    return model


# ============================================================
# MODEL TEST
# ============================================================

def test_model(
    device=None,
    sequence_length=64,
):
    """
    Basic forward test.
    """

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = create_model(
        device=device
    )

    model.eval()

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (
            1,
            sequence_length,
        ),
        device=device,
    )

    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (
            1,
            sequence_length,
        ),
        device=device,
    )

    with torch.no_grad():

        logits, loss = model(
            input_ids,
            targets,
        )

    return {
        "model": model,
        "input_shape":
            tuple(input_ids.shape),
        "logits_shape":
            tuple(logits.shape),
        "loss":
            float(loss),
        "parameters":
            model.parameter_count(),
    }


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("VENLA V0.1 - MODEL TEST")
    print("=" * 60)

    print()

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model = create_model(
        device=device
    )

    print()

    print(
        "Parameters:",
        f"{model.parameter_count():,}",
    )

    result = test_model(
        device=device
    )

    print()

    print(
        "Input shape:",
        result["input_shape"],
    )

    print(
        "Logits shape:",
        result["logits_shape"],
    )

    print(
        "Loss:",
        result["loss"],
    )

    print()

    print("=" * 60)
    print("VENLA MODEL TEST SELESAI")
    print("=" * 60)
'''


MODEL_PATH = os.path.join(
    VENLA_DIR,
    "model.py"
)


with open(
    MODEL_PATH,
    "w",
    encoding="utf-8"
) as f:
    f.write(MODEL_CODE)


print("=" * 60)
print("VENLA MODEL SOURCE CREATED")
print("=" * 60)
print()
print("File:")
print(MODEL_PATH)
print()
print("Size:")
print(
    os.path.getsize(MODEL_PATH),
    "bytes"
)
print()
print("Isi utama:")
print("  VENLA V0.1")
print("  Vocabulary : 32,768")
print("  Context    : 512")
print("  Embedding  : 768")
print("  Layers     : 9")
print("  Heads      : 12")
print("  MLP        : 3,584")
print()
print("✅ model.py berhasil dibuat.")
