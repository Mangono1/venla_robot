"""
VENLA V0.1
Tokenizer V1

Tokenizer karakter sederhana untuk tahap awal VENLA.

Fitur:
- Vocabulary 32,768
- encode
- decode
- BOS token
- EOS token
- PAD token
- UNK token
- save JSON
- load JSON
- validasi vocabulary
"""

import json
import os


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ============================================================
# TOKENIZER
# ============================================================

class VENLATokenizer:
    """
    VENLA Tokenizer V1.

    Tahap awal menggunakan pemetaan karakter Unicode
    ke ID token.

    Untuk karakter Unicode umum:

        token_id = ord(character) + 4

    Special token menggunakan ID 0-3.

    Vocabulary maksimum:
        32,768
    """

    def __init__(
        self,
        vocab_size=32768,
    ):

        self.vocab_size = int(
            vocab_size
        )

        if self.vocab_size < 256:

            raise ValueError(
                "vocab_size terlalu kecil."
            )

        self.special_tokens = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }

        self.id_to_special = {
            value: key
            for key, value
            in self.special_tokens.items()
        }


    # ========================================================
    # CHARACTER TO ID
    # ========================================================

    def char_to_id(
        self,
        character,
    ):
        """
        Convert one character to token ID.
        """

        if not character:

            raise ValueError(
                "Character kosong."
            )

        codepoint = ord(
            character
        )

        token_id = codepoint + 4

        if token_id >= self.vocab_size:

            return UNK_ID

        return token_id


    # ========================================================
    # ID TO CHARACTER
    # ========================================================

    def id_to_char(
        self,
        token_id,
    ):
        """
        Convert token ID back to character.
        """

        token_id = int(
            token_id
        )

        if token_id in self.id_to_special:

            return self.id_to_special[
                token_id
            ]

        codepoint = token_id - 4

        if codepoint < 0:

            return UNK_TOKEN

        try:

            return chr(
                codepoint
            )

        except ValueError:

            return UNK_TOKEN


    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False,
    ):
        """
        Convert text into token IDs.
        """

        if not isinstance(
            text,
            str,
        ):

            raise TypeError(
                "text harus berupa string."
            )

        tokens = []

        if add_bos:

            tokens.append(
                BOS_ID
            )

        for character in text:

            tokens.append(
                self.char_to_id(
                    character
                )
            )

        if add_eos:

            tokens.append(
                EOS_ID
            )

        return tokens


    # ========================================================
    # DECODE
    # ========================================================

    def decode(
        self,
        tokens,
        skip_special_tokens=True,
    ):
        """
        Convert token IDs back into text.
        """

        output = []

        for token_id in tokens:

            token_id = int(
                token_id
            )

            if (
                skip_special_tokens
                and
                token_id in self.id_to_special
            ):

                continue

            character = self.id_to_char(
                token_id
            )

            if (
                character
                in self.special_tokens
            ):

                if skip_special_tokens:

                    continue

            output.append(
                character
            )

        return "".join(
            output
        )


    # ========================================================
    # VOCABULARY
    # ========================================================

    def get_vocab_size(self):

        return self.vocab_size


    # ========================================================
    # SPECIAL TOKEN IDS
    # ========================================================

    def get_pad_id(self):

        return PAD_ID


    def get_unk_id(self):

        return UNK_ID


    def get_bos_id(self):

        return BOS_ID


    def get_eos_id(self):

        return EOS_ID


    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path,
    ):
        """
        Save tokenizer configuration.
        """

        directory = os.path.dirname(
            os.path.abspath(path)
        )

        os.makedirs(
            directory,
            exist_ok=True
        )

        data = {
            "tokenizer_name":
                "VENLA Tokenizer",

            "version":
                "V1",

            "type":
                "character",

            "vocab_size":
                self.vocab_size,

            "special_tokens":
                self.special_tokens,
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return path


    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path,
    ):
        """
        Load tokenizer configuration.
        """

        if not os.path.exists(path):

            raise FileNotFoundError(
                "Tokenizer tidak ditemukan: "
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

        tokenizer = cls(
            vocab_size=int(
                data[
                    "vocab_size"
                ]
            )
        )

        return tokenizer


    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self):

        assert (
            self.vocab_size
            >= 32768
        )

        assert (
            PAD_ID
            == 0
        )

        assert (
            UNK_ID
            == 1
        )

        assert (
            BOS_ID
            == 2
        )

        assert (
            EOS_ID
            == 3
        )

        return True


    # ========================================================
    # INFORMATION
    # ========================================================

    def info(self):

        return {
            "name":
                "VENLA Tokenizer",

            "version":
                "V1",

            "type":
                "character",

            "vocab_size":
                self.vocab_size,

            "pad_id":
                PAD_ID,

            "unk_id":
                UNK_ID,

            "bos_id":
                BOS_ID,

            "eos_id":
                EOS_ID,
        }


# ============================================================
# FACTORY
# ============================================================

def create_tokenizer():

    return VENLATokenizer(
        vocab_size=32768
    )


# ============================================================
# TEST
# ============================================================

def test_tokenizer():

    print("=" * 60)
    print("VENLA V0.1 - TOKENIZER V1")
    print("=" * 60)

    print()

    tokenizer = create_tokenizer()

    tokenizer.validate()

    print(
        "Vocabulary:",
        tokenizer.get_vocab_size()
    )

    print()

    text = (
        "Halo, saya VENLA. "
        "Kita membuat model dari nol."
    )

    print(
        "Text:",
        text
    )

    print()

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    print(
        "Token count:",
        len(tokens)
    )

    print()

    print(
        "Tokens:",
        tokens
    )

    print()

    decoded = tokenizer.decode(
        tokens
    )

    print(
        "Decoded:",
        decoded
    )

    print()

    assert decoded == text

    print(
        "✅ ENCODE/DECODE BERHASIL"
    )

    print()

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:

        path = os.path.join(
            temp_dir,
            "venla_tokenizer.json",
        )

        tokenizer.save(
            path
        )

        print(
            "Tokenizer saved:",
            path
        )

        loaded = VENLATokenizer.load(
            path
        )

        loaded.validate()

        encoded_again = loaded.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        decoded_again = loaded.decode(
            encoded_again
        )

        assert encoded_again == tokens

        assert decoded_again == text

    print(
        "✅ SAVE/LOAD BERHASIL"
    )

    print()

    print("=" * 60)
    print("TOKENIZER V1 SELESAI")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_tokenizer()
