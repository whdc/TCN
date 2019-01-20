from torch import nn
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_labels, num_channels, kernel_sizes, dropout=0.2, emb_dropout=0.2, pool=False):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_sizes=kernel_sizes, dropout=dropout, pool=pool)
        self.auto_decoder = nn.Linear(input_size, output_size)
        self.auto_decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

        self.label_decoder = nn.Linear(input_size, n_labels)
        self.label_decoder.weight.data.uniform_(-0.1, 0.1)
        self.label_decoder.bias.data.fill_(0)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.auto_decoder.bias.data.fill_(0)
        self.auto_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y = self.tcn(emb.transpose(1, 2))
        auto = self.auto_decoder(y.transpose(1, 2))
        label = self.label_decoder(y.transpose(1, 2))
        return auto.contiguous(), label.contiguous()
