import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer

MODEL_CHECKPOINT = "bert-base-uncased"
VOCAB_SIZE = 30522
HIDDEN_SIZE = 768
MAX_POSITION = 512
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072
DROPOUT_P = 0.1


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_position):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_position, embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_len = input_ids.size(-1)
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, -1)

        token_embed = self.token_embeddings(input_ids)
        position_embed = self.position_embeddings(position_ids)

        embeddings = token_embed + position_embed
        embeddings = self.layernorm(embeddings)
        return self.dropout(embeddings)


class Attention(nn.Module):
    def __init__(self, embed_dim, head_dim, mask=None):
        super().__init__()
        self.mask = mask
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        d_k = q.size(-1)  # head_dim
        scores = torch.bmm(q, k.transpose(1, 2)) / d_k ** 0.5

        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, -float("inf"))

        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mask=None):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            Attention(embed_dim, self.head_dim, mask) for _ in range(num_heads)
        ])
        self.out_fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.out_fc(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout_p):
        super().__init__()
        self.linear_in = nn.Linear(embed_dim, hidden_size)
        self.linear_out = nn.Linear(hidden_size, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.gelu(x)
        x = self.linear_out(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, intermediate_size, dropout_p):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, intermediate_size, dropout_p)

    def forward(self, x):
        x = x + self.mha(self.layernorm_1(x))
        x = x + self.ff(self.layernorm_2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, intermediate_size, dropout_p, seq_len):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.layernorm_3 = nn.LayerNorm(embed_dim)
        self.mask = torch.tril(torch.ones(seq_len, seq_len))
        self.mha_1 = MultiHeadAttention(embed_dim, num_heads, mask=self.mask)
        self.mha_2 = MultiHeadAttention(embed_dim, num_heads, mask=self.mask)
        self.ff = FeedForward(embed_dim, intermediate_size, dropout_p)

    def forward(self, x):
        x = x + self.mha_1(self.layernorm_1(x))
        x = x + self.mha_2(self.layernorm_2(x))
        x = x + self.ff(self.layernorm_3(x))
        return x


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    embed = Embeddings(VOCAB_SIZE, HIDDEN_SIZE, MAX_POSITION)
    inputs_embedded = embed(inputs.input_ids)

    encoder = EncoderBlock(HIDDEN_SIZE, NUM_HEADS,
                           INTERMEDIATE_SIZE, DROPOUT_P)
    encoder_output = encoder(inputs_embedded)

    seq_len = inputs_embedded.size(-2)
    decoder = DecoderBlock(HIDDEN_SIZE, NUM_HEADS,
                           INTERMEDIATE_SIZE, DROPOUT_P, seq_len)
    decoder_output = decoder(encoder_output)

    assert decoder_output.size() == encoder_output.size()
    assert decoder_output.size() == inputs_embedded.size()
