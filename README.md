# VENLA

VENLA adalah proyek model bahasa dan autonomous intelligence framework
yang dikembangkan dari nol menggunakan Python dan PyTorch.

Repository ini menjadi pusat kode sumber VENLA.

---

## VENLA V0.1

Versi awal berfokus pada pembangunan model bahasa
decoder-only Transformer sekitar 100 juta parameter.

### Arsitektur

- Vocabulary: 32,768
- Context Length: 512
- Embedding: 768
- Transformer Layers: 9
- Attention Heads: 12
- MLP Dimension: 3,584
- Architecture: Decoder-only Transformer
- Normalization: RMSNorm
- Attention: Causal Self Attention
- Language Modeling: Causal Language Modeling

Target parameter:

~96 juta parameter.

---

# Repository Structure

```text
venla_robot/
│
├── venla/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── checkpoint.py
│   ├── supabase.py
│   └── train.py
│
├── data/
│   └── train.txt
│
├── artifacts/
│   ├── tokenizer/
│   ├── checkpoints/
│   ├── configs/
│   └── logs/
│
├── colab_train.py
├── requirements.txt
└── README.md
