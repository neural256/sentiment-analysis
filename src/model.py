import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_sequence_length: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(max_sequence_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)

        if embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        sequence_length = x.size(1)
        x = x + self.pe[:, :sequence_length, :]
        return self.dropout(x)


class TransformerSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        feedforward_dim: int,
        num_classes: int,
        max_sequence_length: int,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids):
        padding_mask = input_ids.eq(self.pad_idx)

        x = self.embedding(input_ids)
        x = x * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)

        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        cls_representation = encoded[:, 0, :]

        x = self.dropout(cls_representation)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        logits = self.output(x)

        return logits
