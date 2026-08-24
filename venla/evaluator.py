"""
VENLA V0.1
Evaluator Engine V1

Fungsi:
- evaluasi validation loss
- perplexity
- sample prediction
- token accuracy
- model evaluation tanpa gradient
"""

import math
import torch


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# EVALUATOR
# ============================================================

class VENLAEvaluator:
    """
    Evaluator untuk model language modeling VENLA.
    """

    def __init__(
        self,
        model,
        device=None,
    ):

        self.model = model

        self.device = (
            device
            if device is not None
            else get_device()
        )

        self.model.to(
            self.device
        )


    # ========================================================
    # EVALUATE
    # ========================================================

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        max_batches=None,
    ):

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        total_correct = 0
        total_tokens = 0

        for batch_index, batch in enumerate(
            dataloader
        ):

            if (
                max_batches is not None
                and
                batch_index >= max_batches
            ):

                break

            input_ids, targets = batch

            input_ids = input_ids.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            logits, loss = self.model(
                input_ids,
                targets,
            )


            total_loss += float(
                loss.detach().cpu()
            )

            total_batches += 1


            # ------------------------------------------------
            # ACCURACY
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            correct = (
                predictions
                == targets
            ).sum().item()

            total_correct += correct

            total_tokens += (
                targets.numel()
            )


        if total_batches == 0:

            raise RuntimeError(
                "Tidak ada batch untuk evaluasi."
            )


        average_loss = (
            total_loss
            / total_batches
        )


        accuracy = (
            total_correct
            / total_tokens
            if total_tokens > 0
            else 0.0
        )


        # ----------------------------------------------------
        # PERPLEXITY
        # ----------------------------------------------------

        try:

            perplexity = math.exp(
                average_loss
            )

        except OverflowError:

            perplexity = float(
                "inf"
            )


        result = {
            "loss":
                average_loss,

            "perplexity":
                perplexity,

            "accuracy":
                accuracy,

            "accuracy_percent":
                accuracy * 100.0,

            "batches":
                total_batches,

            "tokens":
                total_tokens,
        }


        return result


    # ========================================================
    # PRINT EVALUATION
    # ========================================================

    def print_result(
        self,
        result,
    ):

        print("=" * 60)
        print("VENLA EVALUATION")
        print("=" * 60)

        print()

        print(
            "Loss:",
            f"{result['loss']:.6f}"
        )

        print(
            "Perplexity:",
            f"{result['perplexity']:.4f}"
        )

        print(
            "Accuracy:",
            f"{result['accuracy_percent']:.4f}%"
        )

        print(
            "Batches:",
            result["batches"]
        )

        print(
            "Tokens:",
            result["tokens"]
        )

        print()


    # ========================================================
    # PREDICT NEXT TOKEN
    # ========================================================

    @torch.no_grad()
    def predict_next_token(
        self,
        input_ids,
    ):

        self.model.eval()

        if not isinstance(
            input_ids,
            torch.Tensor,
        ):

            input_ids = torch.tensor(
                input_ids,
                dtype=torch.long,
            )


        if input_ids.dim() == 1:

            input_ids = input_ids.unsqueeze(
                0
            )


        input_ids = input_ids.to(
            self.device
        )


        logits, _ = self.model(
            input_ids
        )


        last_logits = logits[
            :,
            -1,
            :
        ]


        next_token = torch.argmax(
            last_logits,
            dim=-1,
        )


        return next_token


    # ========================================================
    # TOP-K NEXT TOKENS
    # ========================================================

    @torch.no_grad()
    def top_k_tokens(
        self,
        input_ids,
        k=10,
    ):

        self.model.eval()

        if not isinstance(
            input_ids,
            torch.Tensor,
        ):

            input_ids = torch.tensor(
                input_ids,
                dtype=torch.long,
            )


        if input_ids.dim() == 1:

            input_ids = input_ids.unsqueeze(
                0
            )


        input_ids = input_ids.to(
            self.device
        )


        logits, _ = self.model(
            input_ids
        )


        last_logits = logits[
            :,
            -1,
            :
        ]


        probabilities = torch.softmax(
            last_logits,
            dim=-1,
        )


        values, indices = torch.topk(
            probabilities,
            k=k,
            dim=-1,
        )


        return (
            indices,
            values,
        )


# ============================================================
# PERPLEXITY HELPER
# ============================================================

def calculate_perplexity(
    loss,
):

    loss = float(
        loss
    )

    try:

        return math.exp(
            loss
        )

    except OverflowError:

        return float(
            "inf"
        )


# ============================================================
# EVALUATION TEST
# ============================================================

def test_evaluator():

    print("=" * 60)
    print("VENLA V0.1 - EVALUATOR TEST")
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
        "yang belajar memprediksi "
        "token berikutnya. "
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
        shuffle=False,
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
    # EVALUATOR
    # --------------------------------------------------------

    evaluator = VENLAEvaluator(
        model=model,
        device=device,
    )


    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    result = evaluator.evaluate(
        dataloader,
        max_batches=2,
    )


    evaluator.print_result(
        result
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    input_ids, targets = (
        engine.get_sample(0)
    )


    next_token = (
        evaluator.predict_next_token(
            input_ids
        )
    )


    print(
        "Predicted next token:",
        next_token.tolist()
    )


    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    indices, probabilities = (
        evaluator.top_k_tokens(
            input_ids,
            k=5,
        )
    )


    print()

    print(
        "Top-5 token IDs:",
        indices[0].tolist()
    )


    print(
        "Top-5 probabilities:",
        [
            round(
                float(value),
                6,
            )
            for value
            in probabilities[0]
        ]
    )


    print()

    # --------------------------------------------------------
    # PERPLEXITY
    # --------------------------------------------------------

    ppl = calculate_perplexity(
        result["loss"]
    )


    assert abs(
        ppl
        - result["perplexity"]
    ) < 1e-6


    print(
        "Perplexity check:",
        ppl
    )


    print()

    print("=" * 60)
    print("✅ EVALUATOR ENGINE TEST SELESAI")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_evaluator()
