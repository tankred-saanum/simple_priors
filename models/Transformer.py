import math
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset




class PositionalEncoding(nn.Module):
    def __init__(self, input_dims, max_len):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dims, 2) * (-math.log(10000.0) / input_dims))
        pe = torch.zeros(max_len, 1, input_dims)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        #self.pe = pe#nn.Parameter(pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x
        #return self.dropout(x)

class ActionTransformer(nn.Module):
    def __init__(self, action_dims, input_dims, output_dims, nhead, hidden_dims, nlayers, max_len):
        super(ActionTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_encoder = PositionalEncoding(input_dims, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(input_dims, nhead, hidden_dims)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.input_dims = input_dims
        self.encoder = nn.Sequential(
                        nn.Linear(action_dims, hidden_dims),
                        nn.ReLU(),
                        nn.Linear(hidden_dims, input_dims)
                        )


        # decoder_layers = TransformerDecoderLayer(input_dims, nhead, hidden_dims)
        # self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        #self.encoder = nn.Linear(action_dims, input_dims)
        self.decoder_mu = nn.Linear(input_dims, output_dims)
        self.decoder_logstd = nn.Linear(input_dims, output_dims)



    def encode(self, src):
        src_mask = self.generate_square_subsequent_mask(src.size(0))
        src = self.encoder(src) * math.sqrt(self.input_dims)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output

    #
    # def encode_autoregressive(self, next_token):
    #     src_mask = self.generate_square_subsequent_mask(self.z_t.size(0)+1)
    #     z_next = self.encoder(next_token) * math.sqrt(self.input_dims)
    #     z_next = self.o


    def forward(self, input):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src_mask = self.generate_square_subsequent_mask(input.size(0))
        src = self.encoder(input) * math.sqrt(self.input_dims)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)

        #output = self.transformer_decoder(input, encoding, src_mask, src_mask)



        mu = self.decoder_mu(output)
        log_std = self.decoder_logstd(output)
        return mu, log_std

    def decode(self, encoding):
        mu = self.decoder_mu(encoding)
        log_std = self.decoder_logstd(encoding)
        return mu, log_std


    def generate_square_subsequent_mask(self, sequence_length):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sequence_length, sequence_length) * float('-inf'), diagonal=1).to(device=self.device)

    def generate_causal_mask(self, sequence_length):
        return (torch.ones(sequence_length, sequence_length).tril()==0).to(device=self.device)




class ActionTransformerDiscrete(ActionTransformer):
    def __init__(self, action_dims, input_dims, output_dims, nhead, hidden_dims, nlayers, max_len):
        super(ActionTransformerDiscrete, self).__init__(action_dims, input_dims, output_dims, nhead, hidden_dims, nlayers, max_len)

        self.decoder = nn.Linear(input_dims, output_dims)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src_mask = self.generate_square_subsequent_mask(src.size(0))
        src = self.encoder(src) * math.sqrt(self.input_dims)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        probs = torch.softmax(self.decoder(output), dim=-1)
        return probs


class VariationalActionTransformerDiscrete(ActionTransformer):
    def __init__(self, action_dims, input_dims, output_dims, nhead, hidden_dims, nlayers, max_len):
        super(ActionTransformerDiscrete, self).__init__(action_dims, input_dims, output_dims, nhead, hidden_dims, nlayers, max_len)

        self.decoder = nn.Linear(input_dims, output_dims)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src_mask = self.generate_square_subsequent_mask(src.size(0))
        src = self.encoder(src) * math.sqrt(self.input_dims)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        probs = torch.softmax(self.decoder(output), dim=-1)
        return probs


    def encode(self, src):
        src_mask = self.generate_square_subsequent_mask(src.size(0))
        src = self.encoder(src) * math.sqrt(self.input_dims)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output





# model = ActionTransformer(action_dims=4, input_dims=8, output_dims=6, nhead=2, hidden_dims=256, nlayers=2)
#
# x = torch.randn(10, 5, 4)
# y = model(x)
# y.shape
#
#
# model.mask
