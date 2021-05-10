import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalReconstructor(nn.Module):
    def __init__(self, model_name, n_layers, decoder_hidden_size, hidden_size, dropout, decoder_dropout, attn_size):
        super(LocalReconstructor, self).__init__()
        self.model_name = model_name
        self.n_layers = n_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.decoder_dropout_p = decoder_dropout
        self.attn_size = attn_size

        """
        W,U,B,w is the parameters of the Attention Model
        """
        self.attn_W = nn.Linear(self.hidden_size, self.attn_size, bias=False)
        self.attn_U = nn.Linear(self.decoder_hidden_size, self.attn_size, bias=False)
        self.attn_b = nn.Parameter(torch.ones(self.attn_size), requires_grad=True)
        """
        attn_tanh and attn_softmax is the postprocess of the Attention Model
        """
        self.attn_tanh = nn.Tanh()
        self.attn_w = nn.Linear(self.attn_size, 1, bias=False)
        self.attn_softmax = nn.Softmax()

        self.decoder_dropout = nn.Dropout(self.decoder_dropout_p)
        if self.model_name == "LSTM":
            rnn_unit = nn.LSTM
        else:
            rnn_unit = nn.GRU
        """
        Get the RNN as Reconstructor
        """
        self.rnn = rnn_unit(
            input_size=self.decoder_hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, decoder_hiddens):
        """
        :param hidden:hidden states of the encoder
        :param decoder_hiddens:hidden states of the decoder
        :return:output of the model and update the hidden
        """
        """
        Use the attention to process the hidden
        """
        if self.model_name == "LSTM":
            Wh = self.attn_W(hidden[0][-1])
        else:
            Wh = self.attn_W(hidden[-1])
        Uv = self.attn_U(decoder_hiddens)
        Wh = Wh.unsqueeze(0).unsqueeze(0).expand_as(Uv)
        betas = Wh + Uv + self.attn_b
        betas = self.attn_tanh(betas)
        betas = self.attn_w(betas)

        """
        input the features to the rnn.
        """
        betas = betas.expand_as(decoder_hiddens)
        weighted_decoder_hiddens = betas * decoder_hiddens
        input = weighted_decoder_hiddens.mean(dim=0)
        input = self.decoder_dropout(input)
        output, hidden = self.rnn(input, hidden)

        output = self.out(output[0])
        return output, hidden


class GlobalReconstructor(nn.Module):
    def __init__(self, model_name, n_layers, decoder_hidden_size, hidden_size, dropout, decoder_dropout, caption_max_len):
        super(GlobalReconstructor, self).__init__()
        self.model_name = model_name
        self.n_layers = n_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.decoder_dropout_p = decoder_dropout
        self.caption_max_len = caption_max_len

        self.decoder_dropout = nn.Dropout(self.decoder_dropout_p)
        if self.model_name == "LSTM":
            rnn_unit = nn.LSTM
        else:
            rnn_unit = nn.GRU
        self.rnn = rnn_unit(
            input_size=self.decoder_hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, decoder_hiddens):
        batch_specific_len = decoder_hiddens.size()[0]
        """
        GlobalReconstructor is similar as LocalReconstructor but without attention.
        """
        mean_pooled = decoder_hiddens.transpose(0, 2)
        mean_pooled = mean_pooled.transpose(1, 3)
        mean_pooled = mean_pooled.mean(2)
        mean_pooled = mean_pooled.mean(2)
        mean_pooled = mean_pooled / batch_specific_len * self.caption_max_len
        mean_pooled = self.decoder_dropout(mean_pooled)

        input_combined = torch.cat((input[0], mean_pooled), 1)
        input_combined = input_combined.unsqueeze(0)

        output, hidden = self.rnn(input_combined, hidden)

        output = self.out(output[0])
        return output, hidden


