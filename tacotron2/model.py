# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import sys
from os.path import abspath, dirname
# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/../'))
from common.layers import ConvNorm, LinearNorm
from common.utils import to_gpu, get_mask_from_lengths


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x, inference=False):
        if inference:
            for linear in self.layers:
                x = F.relu(linear(x))
                x0 = x[0].unsqueeze(0)
                mask = Variable(torch.bernoulli(x0.data.new(x0.data.size()).fill_(0.5)))
                mask = mask.expand(x.size(0), x.size(1))
                x = x*mask*2
        else:
            for linear in self.layers:
                x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, n_mel_channels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_n_convolutions,
                 encoder_embedding_dim, encoder_kernel_size):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim))
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(encoder_embedding_dim,
                            int(encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, n_mel_channels, n_frames_per_step,
                 encoder_embedding_dim, attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size,
                 attention_rnn_dim, decoder_rnn_dim,
                 prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 early_stopping):
        super(Decoder, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = Prenet(
            n_mel_channels * n_frames_per_step,
            [prenet_dim, prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim)

        self.attention_layer = Attention(
            attention_rnn_dim, encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels * n_frames_per_step)

        self.gate_layer = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        #print('Len Alignments:', len(alignments))
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        #print('Len Gate Outputs:', len(gate_outputs))

        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze() if memory.shape[0] > 1 else gate_output]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def infer(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32).cuda()
        not_finished = torch.ones([memory.size(0)], dtype=torch.int32).cuda()
        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]
            dec = torch.le(torch.sigmoid(gate_output.data),
                           self.gate_threshold).to(torch.int32).squeeze(1)

            not_finished = not_finished*dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class GST(nn.Module):
    def __init__(self, n_mel_channels, style_embedding_dim,
                 gst_n_tokens, gst_n_heads, ref_enc_filters, ref_enc_kernel_size,
                 ref_enc_stride, ref_enc_pad, ref_enc_gru_dim):
        super(GST, self).__init__()

        self.encoder = ReferenceEncoder(n_mel_channels, ref_enc_filters, ref_enc_kernel_size,
                                        ref_enc_stride, ref_enc_pad, ref_enc_gru_dim)

        self.stl = STL(style_embedding_dim, gst_n_tokens, gst_n_heads)

    def forward(self, inputs):                 # [N, Ty/r, n_mels*r] (r=n_frames_per_step)
        enc_out = self.encoder(inputs)         # [N, ref_enc_gru_size]
        style_embed = self.stl(enc_out)        # [N, 1, style_embedding_size]

        return style_embed


class ReferenceEncoder(nn.Module):
    """
    inputs  [N, Ty/r, n_mels*r]
    outputs [N, ref_enc_gru_size]
    """
    def __init__(self, n_mel_channels, ref_enc_filters, ref_enc_kernel_size,
                 ref_enc_stride, ref_enc_pad, ref_enc_gru_dim):
        super(ReferenceEncoder, self).__init__()

        self.n_mel_channels = n_mel_channels

        n_filters = len(ref_enc_filters)
        filters = [1] + ref_enc_filters

        convolutions = []

        out_channels = self._calculate_channels(self.n_mel_channels, ref_enc_kernel_size,
                                                ref_enc_stride, ref_enc_pad, n_filters)

        for i in range(n_filters):
            conv_layer = nn.Sequential(
                    nn.Conv2d(in_channels=filters[i],
                          out_channels=filters[i + 1],
                          kernel_size=(ref_enc_kernel_size, ref_enc_kernel_size),
                          stride=(ref_enc_stride, ref_enc_stride),
                          padding=(ref_enc_pad, ref_enc_pad)
                        ),
                    nn.BatchNorm2d(num_features=ref_enc_filters[i])
                )
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=ref_enc_gru_dim,
                          batch_first=True)

    def _calculate_channels(self, S, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            S = (S - kernel_size + 2 * pad) // stride + 1
        return S

    def forward(self, inputs):
        N = inputs.size(0)
        x = inputs.view(N, 1, -1, self.n_mel_channels) # [N, 1, Ty, n_mel_channels]

        for conv in self.convolutions:                 # [N, 128, Ty//2^n_filters, n_mel_channels//2^n_filters]
            x = F.relu(conv(x))

        x = x.transpose(1, 2)                          # [N, Ty//2^n_filters, 128, n_mel_channels//2^n_filters]
        N = x.size(0)
        T = x.size(1)

        x = x.contiguous().view(N, T, -1)              # [N, Ty//2^n_filters, 128*n_mel_channels//2^n_filters]

        memory, x = self.gru(x)                        # [1, N, style_embedding_size//2]

        x = x.squeeze(0)                               # [N, style_embedding_size//2]

        return x


class STL(nn.Module):
    """
    inputs  [N, style_embedding_size//2]
    outputs [N, style_embedding_size]
    """
    def __init__(self, style_embedding_dim, gst_n_tokens, gst_n_heads):
        super(STL, self).__init__()

        d_q = style_embedding_dim // 2
        d_k = style_embedding_dim // gst_n_heads

        self.embed = nn.Parameter(torch.FloatTensor(gst_n_tokens, style_embedding_dim // gst_n_heads))
        nn.init.normal_(self.embed, mean=0, std=0.5)

        self.attention = MultiHeadAttention(d_q, d_k, style_embedding_dim, gst_n_heads)

    def forward(self, inputs):
        N = inputs.size(0)

        queries = inputs.unsqueeze(1)                          # [N, 1, style_embedding_size//2]
        keys = F.tanh(self.embed)
        keys = keys.unsqueeze(0).expand(N, -1, -1)             # [N, gst_n_tokens, style_embedding_size//gst_n_heads]

        style_embedding = self.attention(queries, keys, keys)  # [N, 1, style_embedding_size]

        return style_embedding


class MultiHeadAttention(nn.Module):
    """
    input:
        query [N, T_q, query_dim]
        key   [N, T_k, key_dim]
    output:
        out   [N, T_q, num_units]
    """
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super(MultiHeadAttention, self).__init__()

        if key_dim % num_heads != 0:
            raise ValueError("Key depth {} must be divisible by the number of "
                             "attention heads {}".format(key_dim, num_heads))

        self.num_units = num_units
        self.num_heads = num_heads
        self.query_scale = (key_dim//num_heads)**-0.5

        self.query_linear = nn.Linear(query_dim, num_units, bias=False)
        self.key_linear = nn.Linear(key_dim, num_units, bias=False)
        self.value_linear = nn.Linear(key_dim, num_units, bias=False)
        #self.output_linear = nn.Linear(key_dim, output_depth, bias=False)
        #self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_len, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_len, depth / num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")

        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_len, depth/num_heads]
        Output:
            A Tensor with shape [batch_size, seq_len, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape

        return x.permute(0, 2, 1, 3).contiguous().\
                view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values):
        # Linear for Each
        queries = self.query_linear(queries)                     # [N, T_q, num_units]
        keys = self.key_linear(keys)                             # [N, T_k, num_units]
        values = self.value_linear(values)                       # [N, T_k, num_units]

        # Split into heads
        queries = self._split_heads(queries)                     # [N, num_heads, T_q, num_units/num_heads]
        keys = self._split_heads(keys)                           # [N, num_heads, T_k, num_units/num_heads]
        values = self._split_heads(values)                       # [N, num_heads, T_k, num_units/num_heads]

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2)) # [N, num_heads, T_q, T_k]

        # Convert to probabilities
        scores = F.softmax(logits, dim=-1)                       # [N, num_heads, T_q, T_k]

        # Dropout
        #scores = self.dropout(scores)

        # Combine with values to get context
        contexts = torch.matmul(scores, values)                  # [N, num_heads, T_q, num_units/num_heads]

        # Merge heads
        contexts = self._merge_heads(contexts)                   # [N, T_q, num_units]

        # Linear to get output
        # outputs = self.output_linear(contexts)
        return contexts


class Tacotron2(nn.Module):
    def __init__(self, mask_padding, n_mel_channels,
                 n_symbols, symbols_embedding_dim, n_speakers, speakers_embedding_dim,
                 encoder_kernel_size, encoder_n_convolutions, encoder_embedding_dim,
                 attention_rnn_dim, attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, n_frames_per_step,
                 decoder_rnn_dim, prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 postnet_embedding_dim, postnet_kernel_size,
                 postnet_n_convolutions, decoder_no_early_stopping,
                 gst_use, gst_n_tokens=None, gst_n_heads=None, style_embedding_dim=None, ref_enc_filters=None,
                 ref_enc_kernel_size=None, ref_enc_stride=None, ref_enc_pad=None, ref_enc_gru_dim=None, **kwargs):
        super(Tacotron2, self).__init__()
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.gst_use = gst_use

        self.symbols_embedding = nn.Embedding(
            n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.symbols_embedding.weight.data.uniform_(-val, val)

        self.speakers_embedding = nn.Embedding(
            n_speakers, speakers_embedding_dim)

        torch.nn.init.xavier_uniform_(self.speakers_embedding.weight)

        self.encoder = Encoder(encoder_n_convolutions,
                               encoder_embedding_dim,
                               encoder_kernel_size)

        encoder_and_speakers_embedding_dim = encoder_embedding_dim + speakers_embedding_dim

        if gst_use:
            encoder_and_speakers_embedding_dim += style_embedding_dim

            self.gst = GST(n_mel_channels, style_embedding_dim,
                           gst_n_tokens, gst_n_heads, ref_enc_filters, ref_enc_kernel_size,
                           ref_enc_stride, ref_enc_pad, ref_enc_gru_dim)

        self.decoder = Decoder(n_mel_channels, n_frames_per_step,
                               encoder_and_speakers_embedding_dim, attention_dim,
                               attention_location_n_filters,
                               attention_location_kernel_size,
                               attention_rnn_dim, decoder_rnn_dim,
                               prenet_dim, max_decoder_steps,
                               gate_threshold, p_attention_dropout,
                               p_decoder_dropout,
                               not decoder_no_early_stopping)

        self.postnet = Postnet(n_mel_channels, postnet_embedding_dim,
                               postnet_kernel_size,
                               postnet_n_convolutions)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_ids = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speaker_ids = to_gpu(speaker_ids).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_ids),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        # Parse inputs
        inputs, input_lengths, targets, max_len, output_lengths, speaker_ids = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        # Define output tokens:
        outputs = []

        # print('Inputs:', inputs.shape)
        # print('Input lengths:', input_lengths.shape)
        # print('Targets:', targets.shape)
        # print('Max len:', max_len)
        # print('Output lengths:', output_lengths.shape)
        # print('Speaker ids:', speaker_ids.shape)

        # Extract symbols embedding
        embedded_inputs = self.symbols_embedding(inputs).transpose(1, 2)
        # Get symbols encoder outputs
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        outputs.append(encoder_outputs)

        # Extract speaker embeddings
        speaker_ids = speaker_ids.unsqueeze(1)
        embedded_speakers = self.speakers_embedding(speaker_ids)
        embedded_speakers = embedded_speakers.expand(-1, max_len, -1)

        outputs.append(embedded_speakers)

        # GST
        if self.gst_use:
            # Get style tokens
            ref_mels = targets
            style_embeddings = self.gst(ref_mels)
            style_embeddings = style_embeddings.expand(-1, max_len, -1)

            outputs.append(style_embeddings)


        # Combine symbols, style, and speaker embeddings
        merged_outputs = torch.cat(outputs, -1)

        mel_outputs, gate_outputs, alignments = self.decoder(
            merged_outputs, targets, memory_lengths=input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def infer(self, inputs, speaker_id, ref_mel=None, style_token=None):
        # Outputs
        outputs = []

        # Get symbols encoder outputs
        embedded_inputs = self.symbols_embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(embedded_inputs)
        outputs.append(encoder_outputs)

        # Get speaker embedding
        speaker_id = speaker_id.unsqueeze(1)
        embedded_speaker = self.speakers_embedding(speaker_id)
        embedded_speaker = embedded_speaker.expand(-1, encoder_outputs.shape[1], -1)
        outputs.append(embedded_speaker)

        # Get style embeddings
        if self.gst_use:
            if ref_mel is not None:
                style_embeddings = self.gst(ref_mel)
                print('style_embeddings', style_embeddings.shape)
                style_embeddings = style_embeddings.expand(-1, encoder_outputs.shape[1], -1)
                print('style_embeddings exp.', style_embeddings.shape)
                outputs.append(style_embeddings)

            elif style_token is not None:
                pass
            else:
                raise

        # Merge embeddings
        merged_outputs = torch.cat(outputs, -1)

        print('merged_outputs', merged_outputs.shape)

        # Decode
        mel_outputs, gate_outputs, alignments = self.decoder.infer(
            merged_outputs)

        # Post
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # Parse
        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
