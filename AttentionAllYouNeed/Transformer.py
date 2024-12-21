import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        attention = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)          
        
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention,V)
        return output
    
    def forward(self,Q,K,V,mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        K = self.W_k(K).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        V = self.W_v(V).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)

        output = self.scaled_dot_product_attention(Q,K,V,mask)

        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        return self.W_o(output)

class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length,d_model)
        position = torch.arange(0,max_seq_length,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    
    def forward(self,x):
        return x + self.pe[:,:x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.feed_forward = PositionWiseFeedForward(d_model,d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,mask=None):
        attention_output = self.self_attention(x,x,x,mask)
        x = self.norm1(x + self.dropout(attention_output))

        ff_dropout = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_dropout))

        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.cross_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.feed_forward = PositionWiseFeedForward(d_model,d_ff,dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_output,src_mask=None,tgt_mask=None):
        
        self_attention_output = self.self_attention(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))

        cross_attention_output = self.cross_attention(x,encoder_output,encoder_output,src_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length=5000, dropout=0.1):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab_size,d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size,d_model)
        self.positional_encoding = PositionalEncoding(d_model,max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])

        self.output_layer = nn.Linear(d_model,tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self,source,tgt):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)

        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1,seq_length,seq_length),diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask

        return source_mask, tgt_mask
    
    def forward(self,source,tgt):
        source_mask, tgt_mask = self.generate_mask(source,tgt)

        source_embedding = self.dropout(self.positional_encoding(self.src_embed(source)))
        tgt_embedding = self.dropout(self.positional_encoding(self.tgt_embed(tgt)))

        enc_output = source_embedding
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output,source_mask)
        
        dec_output = tgt_embedding
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output,enc_output,source_mask,tgt_mask)
        
        output = self.output_layer(dec_output)
        return output

if __name__ == '__main__':
    # Model parameters
    src_vocab_size = 5000 #Source vocabulary size
    tgt_vocab_size = 5000 #Target vocabulary size
    d_model = 512 # Embedding Dimension
    num_heads = 8 # Number of attention heads
    num_layers = 6 # Number of encoder/decoder layers
    d_ff = 2048 # Feed-Forward dimension
    dropout = 0.1

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )

    src = torch.randint(1, src_vocab_size, (32, 20))
    tgt = torch.randint(1, tgt_vocab_size, (32, 15))
    try:
        output = model(src,tgt)
        print(f"Output'shape is: {output.shape}")
        print(output)
    except Exception as e:
        print(f"Error is: {e}")