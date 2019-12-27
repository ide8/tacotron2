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
import torch

from tacotron2.model import Tacotron2
from waveglow.model import WaveGlow

from configs import Config


def get_model(model_name, model_config, to_cuda):
    """ Code chooses a model based on name"""
    if model_name == 'Tacotron2':
        model = Tacotron2(**model_config)
    elif model_name == 'WaveGlow':
        model = WaveGlow(**model_config)
    else:
        raise NotImplementedError(model_name)
    if to_cuda:
        model = model.cuda()
    return model


def get_model_config(model_name):
    """ Code chooses a model based on name"""
    if model_name == 'Tacotron2':
        model_config = dict(
            # optimization
            mask_padding=Config.mask_padding,
            # audio
            n_mel_channels=Config.n_mel_channels,
            # symbols
            n_symbols=Config.n_symbols,
            symbols_embedding_dim=Config.symbols_embedding_dim,
            # speakers
            n_speakers=Config.n_speakers,
            speakers_embedding_dim=Config.speakers_embedding_dim,
            # emotions
            use_emotions=Config.use_emotions,
            n_emotions=Config.n_emotions,
            emotions_embedding_dim=Config.emotions_embedding_dim,
            # encoder
            encoder_kernel_size=Config.encoder_kernel_size,
            encoder_n_convolutions=Config.encoder_n_convolutions,
            encoder_embedding_dim=Config.encoder_embedding_dim,
            # attention
            attention_rnn_dim=Config.attention_rnn_dim,
            attention_dim=Config.attention_dim,
            # attention location
            attention_location_n_filters=Config.attention_location_n_filters,
            attention_location_kernel_size=Config.attention_location_kernel_size,
            # decoder
            n_frames_per_step=Config.n_frames_per_step,
            decoder_rnn_dim=Config.decoder_rnn_dim,
            prenet_dim=Config.prenet_dim,
            max_decoder_steps=Config.max_decoder_steps,
            gate_threshold=Config.gate_threshold,
            p_attention_dropout=Config.p_attention_dropout,
            p_decoder_dropout=Config.p_decoder_dropout,
            # Postnet
            postnet_embedding_dim=Config.postnet_embedding_dim,
            postnet_kernel_size=Config.postnet_kernel_size,
            postnet_n_convolutions=Config.postnet_n_convolutions,
            decoder_no_early_stopping=Config.decoder_no_early_stopping
        )
        return model_config
    elif model_name == 'WaveGlow':
        model_config = dict(
            n_mel_channels=Config.n_mel_channels,
            n_flows=Config.n_flows,
            n_group=Config.n_group,
            n_early_every=Config.n_early_every,
            n_early_size=Config.n_early_size,
            sigma=Config.wg_sigma,
            WN_config=Config.wn_config
        )
        return model_config
    else:
        raise NotImplementedError(model_name)
