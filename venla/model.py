"""
VENLA V0.1
100M Parameter Decoder-Only Transformer

Architecture:
- Vocabulary : 32,768
- Context    : 512
- Embedding  : 768
- Layers     : 9
- Heads      : 12
- MLP        : 3,584

Target:
~96.4M parameters

Model ini dibuat untuk causal language modeling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_VOCAB_SIZE = 32768
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_EMBED_DIM = 768
DEFAULT_NUM_LAYERS = 9
DEFAULT_NUM_HEADS = 12
DEFAULT_MLP_DIM = 3584

DEFAULT_DROPOUT = 0.0


# ============================================================
# RMS NORMALIZATION
# ============================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Lebih sederhana daripada LayerNorm:
        y = x / RMS(x) * weight
    """

    def __init__(
        self,
        dim,
        eps=1e-6,
    ):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )


    def forward(
        self,
        x,
    ):

        original_dtype = x.dtype

        x_float = x.float()

        variance = (
            x_float
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        x_float = (
            x_float
            * torch.rsqrt(
                variance
                + self.eps
            )
        )

        return (
            self.weight
            * x_float.to(
                original_dtype
            )
        )


# ============================================================
# CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Token hanya boleh melihat token sebelumnya
    dan dirinya sendiri.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        dropout=0.0,
    ):

        super().__init__()


        if embed_dim % num_heads != 0:

            raise ValueError(
                "embed_dim harus habis dibagi "
                "num_heads."
            )


        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = (
            embed_dim
            // num_heads
        )

        self.context_length = (
            context_length
        )


        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3,
            bias=False,
        )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )


        # ----------------------------------------------------
        # DROPOUT
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )


        # ----------------------------------------------------
        # CAUSAL MASK
        # ----------------------------------------------------

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )


        self.register_buffer(
            "causal_mask",
            mask.view(
                1,
                1,
                context_length,
                context_length,
            ),
            persistent=False,
        )


    def forward(
        self,
        x,
    ):

        batch_size, seq_len, _ = (
            x.shape
        )


        if seq_len > self.context_length:

            raise ValueError(
                "Sequence length "
                f"{seq_len} melebihi "
                f"context length "
                f"{self.context_length}."
            )


        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        qkv = self.qkv(
            x
        )


        q, k, v = (
            qkv
            .chunk(
                3,
                dim=-1,
            )
        )


        # ----------------------------------------------------
        # HEADS
        # ----------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )


        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )


        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )


        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        if hasattr(
            F,
            "scaled_dot_product_attention",
        ):

            output = (
                F.scaled_dot_product_attention(

                    q,

                    k,

                    v,

                    attn_mask=None,

                    dropout_p=(
                        self.dropout.p
                        if self.training
                        else 0.0
                    ),

                    is_causal=True,
                )
            )

        else:

            scale = (
                1.0
                / math.sqrt(
                    self.head_dim
                )
            )


            attention_scores = (
                torch.matmul(
                    q,
                    k.transpose(
                        -2,
                        -1,
                    ),
                )
                * scale
            )


            mask = (
                self.causal_mask[
                    :,
                    :,
                    :seq_len,
                    :seq_len,
                ]
            )


            attention_scores = (
                attention_scores.masked_fill(
                    ~mask,
                    torch.finfo(
                        attention_scores.dtype
                    ).min,
                )
            )


            attention_weights = (
                F.softmax(
                    attention_scores,
                    dim=-1,
                )
            )


            attention_weights = (
                self.dropout(
                    attention_weights
                )
            )


            output = torch.matmul(
                attention_weights,
                v,
            )


        # ----------------------------------------------------
        # MERGE HEADS
        # ----------------------------------------------------

        output = (
            output
            .transpose(
                1,
                2,
            )
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.embed_dim,
            )
        )


        output = self.out_proj(
            output
        )


        return output


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):
    """
    MLP Transformer block.

    Menggunakan GELU.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        dropout=0.0,
    ):

        super().__init__()


        self.fc1 = nn.Linear(
            embed_dim,
            mlp_dim,
            bias=False,
        )


        self.fc2 = nn.Linear(
            mlp_dim,
            embed_dim,
            bias=False,
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
        self,
        x,
    ):

        x = self.fc1(
            x
        )

        x = F.gelu(
            x,
            approximate="tanh",
        )

        x = self.fc2(
            x
        )

        x = self.dropout(
            x
        )

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block.

        x
        │
        ├── RMSNorm
        │
        ├── Attention
        │
        └── Residual
        │
        ├── RMSNorm
        │
        ├── MLP
        │
        └── Residual
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        context_length,
        dropout=0.0,
    ):

        super().__init__()


        self.norm1 = RMSNorm(
            embed_dim
        )


        self.attention = (
            CausalSelfAttention(

                embed_dim=embed_dim,

                num_heads=num_heads,

                context_length=context_length,

                dropout=dropout,
            )
        )


        self.norm2 = RMSNorm(
            embed_dim
        )


        self.mlp = FeedForward(

            embed_dim=embed_dim,

            mlp_dim=mlp_dim,

            dropout=dropout,
        )


    def forward(
        self,
        x,
    ):

        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        x = (
            x
            + self.attention(
                self.norm1(x)
            )
        )


        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        x = (
            x
            + self.mlp(
                self.norm2(x)
            )
        )


        return x


# ============================================================
# VENLA MODEL
# ============================================================

class VENLA(nn.Module):
    """
    VENLA V0.1.

    Decoder-only Transformer
    untuk causal language modeling.
    """

    def __init__(
        self,
        vocab_size=DEFAULT_VOCAB_SIZE,
        context_length=DEFAULT_CONTEXT_LENGTH,
        embed_dim=DEFAULT_EMBED_DIM,
        num_layers=DEFAULT_NUM_LAYERS,
        num_heads=DEFAULT_NUM_HEADS,
        mlp_dim=DEFAULT_MLP_DIM,
        dropout=DEFAULT_DROPOUT,
    ):

        super().__init__()


        # ----------------------------------------------------
        # CONFIG
        # ----------------------------------------------------

        self.vocab_size = (
            vocab_size
        )

        self.context_length = (
            context_length
        )

        self.embed_dim = (
            embed_dim
        )

        self.num_layers = (
            num_layers
        )

        self.num_heads = (
            num_heads
        )

        self.mlp_dim = (
            mlp_dim
        )

        self.dropout_rate = (
            dropout
        )


        # ----------------------------------------------------
        # TOKEN EMBEDDING
        # ----------------------------------------------------

        self.token_embedding = (
            nn.Embedding(

                vocab_size,

                embed_dim,
            )
        )


        # ----------------------------------------------------
        # POSITION EMBEDDING
        # ----------------------------------------------------

        self.position_embedding = (
            nn.Embedding(

                context_length,

                embed_dim,
            )
        )


        # ----------------------------------------------------
        # EMBEDDING DROPOUT
        # ----------------------------------------------------

        self.embedding_dropout = (
            nn.Dropout(
                dropout
            )
        )


        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        self.layers = nn.ModuleList(

            [

                TransformerBlock(

                    embed_dim=embed_dim,

                    num_heads=num_heads,

                    mlp_dim=mlp_dim,

                    context_length=context_length,

                    dropout=dropout,
                )

                for _ in range(
                    num_layers
                )

            ]

        )


        # ----------------------------------------------------
        # FINAL NORMALIZATION
        # ----------------------------------------------------

        self.final_norm = RMSNorm(
            embed_dim
        )


        # ----------------------------------------------------
        # LANGUAGE MODEL HEAD
        # ----------------------------------------------------

        self.lm_head = nn.Linear(

            embed_dim,

            vocab_size,

            bias=False,
        )


        # ----------------------------------------------------
        # WEIGHT TYING
        # ----------------------------------------------------

        self.lm_head.weight = (
            self.token_embedding.weight
        )


        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        self.apply(
            self._init_weights
        )


        # Re-apply tied weight after
        # initialization.
        self.lm_head.weight = (
            self.token_embedding.weight
        )


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _init_weights(
        self,
        module,
    ):

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
            nn.Embedding,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )


    # ========================================================
    # MODEL CONFIG
    # ========================================================

    def model_config(
        self,
    ):

        return {

            "model_name":
                "VENLA",

            "version":
                "V0.1",

            "architecture":
                "decoder-only-transformer",

            "vocab_size":
                self.vocab_size,

            "context_length":
                self.context_length,

            "embed_dim":
                self.embed_dim,

            "num_layers":
                self.num_layers,

            "num_heads":
                self.num_heads,

            "mlp_dim":
                self.mlp_dim,

            "dropout":
                self.dropout_rate,

            "parameters":
                self.num_parameters(),
        }


    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    def num_parameters(
        self,
    ):

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )


    def num_trainable_parameters(
        self,
    ):

        return sum(

            parameter.numel()

            for parameter
            in self.parameters()

            if parameter.requires_grad
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        input_ids,
        targets=None,
    ):

        if input_ids.dim() != 2:

            raise ValueError(
                "input_ids harus memiliki "
                "shape [batch, sequence]."
            )


        batch_size, seq_len = (
            input_ids.shape
        )


        if seq_len > self.context_length:

            raise ValueError(
                "Sequence length "
                f"{seq_len} melebihi "
                f"context length "
                f"{self.context_length}."
            )


        # ----------------------------------------------------
        # POSITIONS
        # ----------------------------------------------------

        positions = torch.arange(

            seq_len,

            device=input_ids.device,

        ).unsqueeze(
            0
        )


        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        token_embeddings = (
            self.token_embedding(
                input_ids
            )
        )


        position_embeddings = (
            self.position_embedding(
                positions
            )
        )


        x = (
            token_embeddings
            + position_embeddings
        )


        x = self.embedding_dropout(
            x
        )


        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        for layer in self.layers:

            x = layer(
                x
            )


        # ----------------------------------------------------
        # FINAL NORM
        # ----------------------------------------------------

        x = self.final_norm(
            x
        )


        # ----------------------------------------------------
        # LOGITS
        # ----------------------------------------------------

        logits = self.lm_head(
            x
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
                    -1
                ),

            )


        return logits, loss


    # ========================================================
    # GENERATION
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
    ):

        self.eval()


        if input_ids.dim() == 1:

            input_ids = input_ids.unsqueeze(
                0
            )


        for _ in range(
            max_new_tokens
        ):

            # ------------------------------------------------
            # CONTEXT WINDOW
            # ------------------------------------------------

            input_context = (
                input_ids[
                    :,
                    -self.context_length:
                ]
            )


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            logits, _ = self(
                input_context
            )


            logits = logits[
                :,
                -1,
                :
            ]


            # ------------------------------------------------
            # TEMPERATURE
            # ------------------------------------------------

            if temperature <= 0:

                raise ValueError(
                    "temperature harus > 0."
                )


            logits = (
                logits
                / temperature
            )


            # ------------------------------------------------
            # TOP-K
            # ------------------------------------------------

            if top_k is not None:

                top_k = min(
                    int(top_k),
                    logits.size(-1),
                )


                values, _ = torch.topk(
                    logits,
                    top_k,
                )


                minimum = values[
                    :,
                    -1,
                    None
                ]


                logits = torch.where(

                    logits < minimum,

                    torch.full_like(
                        logits,
                        float("-inf"),
                    ),

                    logits,
                )


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            probabilities = F.softmax(
                logits,
                dim=-1,
            )


            # ------------------------------------------------
            # SAMPLE
            # ------------------------------------------------

            next_token = (
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                )
            )


            input_ids = torch.cat(

                [
                    input_ids,
                    next_token,
                ],

                dim=1,
            )


        return input_ids


# ============================================================
# MODEL TEST
# ============================================================

def test_model():

    print("=" * 60)
    print("VENLA V0.1 - MODEL TEST")
    print("=" * 60)

    print()


    device = (

        torch.device("cuda")

        if torch.cuda.is_available()

        else torch.device("cpu")
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


        vram = (

            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (
                1024 ** 3
            )
        )


        print(
            "VRAM:",
            f"{vram:.2f} GB"
        )


    print()


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print(
        "Creating VENLA..."
    )


    model = VENLA()


    model = model.to(
        device
    )


    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    config = (
        model.model_config()
    )


    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    print()


    print(
        "Vocabulary:",
        config["vocab_size"]
    )

    print(
        "Context:",
        config["context_length"]
    )

    print(
        "Embedding:",
        config["embed_dim"]
    )

    print(
        "Layers:",
        config["num_layers"]
    )

    print(
        "Heads:",
        config["num_heads"]
    )

    print(
        "MLP:",
        config["mlp_dim"]
    )

    print()


    parameters = (
        model.num_parameters()
    )


    trainable = (
        model.num_trainable_parameters()
    )


    print(
        "Total parameters:",
        f"{parameters:,}"
    )

    print(
        "Trainable:",
        f"{trainable:,}"
    )


    print()


    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    batch_size = 1

    sequence_length = 64


    input_ids = torch.randint(

        0,

        model.vocab_size,

        (
            batch_size,
            sequence_length,
        ),

        device=device,

    )


    targets = torch.randint(

        0,

        model.vocab_size,

        (
            batch_size,
            sequence_length,
        ),

        device=device,

    )


    print("=" * 60)
    print("FORWARD TEST")
    print("=" * 60)

    print()


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


    # --------------------------------------------------------
    # GENERATION TEST
    # --------------------------------------------------------

    generated = model.generate(

        input_ids[:, :8],

        max_new_tokens=8,

        temperature=1.0,

        top_k=20,
    )


    print(
        "Generation shape:",
        tuple(
            generated.shape
        )
    )


    print(
        "Generation test: OK"
    )


    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (
                1024 ** 3
            )
        )


        reserved = (
            torch.cuda.memory_reserved()
            / (
                1024 ** 3
            )
        )


        print()

        print("=" * 60)
        print("GPU MEMORY")
        print("=" * 60)

        print()

        print(
            "Allocated:",
            f"{allocated:.3f} GB"
        )

        print(
            "Reserved:",
            f"{reserved:.3f} GB"
        )


    # --------------------------------------------------------
    # ASSERT
    # --------------------------------------------------------

    assert logits.shape == (

        batch_size,

        sequence_length,

        model.vocab_size,
    )


    assert loss is not None

    assert math.isfinite(
        float(loss)
    )


    print()

    print("=" * 60)
    print("✅ VENLA MODEL TEST BERHASIL")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_model()
