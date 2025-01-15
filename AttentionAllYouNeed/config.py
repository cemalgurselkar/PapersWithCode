from dataclasses import dataclass

@dataclass
class Config:
    # Model parameters
    src_vocab_size: int = 5000 #Source vocabulary size
    tgt_vocab_size: int = 5000 #Target vocabulary size
    max_seq_length: int = 128
    d_model: int = 512 # Embedding Dimension
    num_heads: int = 8 # Number of attention heads
    num_layers: int = 6 # Number of encoder/decoder layers
    d_ff: int = 2048 # Feed-Forward dimension
    dropout: int = 0.1