import torch
from tacotron2.text import symbols


global symbols


class Hyperparameters():
	# Audio params
	sampling_rate=22050                        # Sampling rate
	filter_length=1024                         # Filter length
	hop_length=256  	                       # Hop (stride) length
	win_length=1024 	                       # Window length
	mel_fmin=0.0 		                       # Minimum mel frequency
	mel_fmax=8000.0                            # Maximum mel frequency
	n_mel_channels=80                          # Number of bins in mel-spectrograms
	max_wav_value=32768.0                      # Maximum audiowave value


	### Tacotron Params
	# Optimization
	mask_padding = False                        # Use mask padding

	# Symbols
	n_symbols = len(symbols)                    # Number of symbols in dictionary
	symbols_embedding_dim = 512                  # Text input embedding dimension

	# Speakers
	n_speakers = 128                            # Number of speakers
	speakers_embedding_dim = 16                  # Speaker embedding dimension

	# Encoder
	encoder_kernel_size = 5                     # Encoder kernel size
	encoder_n_convolutions = 3                  # Number of encoder convolutions
	encoder_embedding_dim = 512                 # Encoder embedding dimension

	# Attention
	attention_rnn_dim = 1024                    # Number of units in attention LSTM
	attention_dim = 128                         # Dimension of attention hidden representation

	# Attention location
	attention_location_n_filters = 32           # Number of filters for location-sensitive attention
	attention_location_kernel_size = 31         # Kernel size for location-sensitive attention

	# Decoder
	n_frames_per_step = 1                       # Number of frames processed per step
	decoder_rnn_dim = 1024                      # Number of units in decoder LSTM
	prenet_dim = 256                            # Number of ReLU units in prenet layers
	max_decoder_steps = 2000                    # Maximum number of output mel spectrograms
	gate_threshold = 0.5                        # Probability threshold for stop token
	p_attention_dropout = 0.1                   # Dropout probability for attention LSTM
	p_decoder_dropout = 0.1                     # Dropout probability for decoder LSTM
	decoder_no_early_stopping = False           # Stop decoding once all samples are finished

	# Postnet
	postnet_embedding_dim = 512                 # Postnet embedding dimension
	postnet_kernel_size = 5                     # Postnet kernel size
	postnet_n_convolutions = 5                  # Number of postnet convolutions

	# GST
	gst_use = True                              # Use or not to use GST
	gst_n_tokens = 10                           # Number of GSTs
	gst_n_heads = 16                            # Number of heads in GST multi-head attention

	# Reference Encoder
	style_embedding_dim = 256                   # Style embedding dim
	ref_enc_filters = [32, 32, 64, 64, 128, 128]# Reference encoder filter
	ref_enc_kernel_size = 3                     # Reference encoder kernel size
	ref_enc_stride = 2                          # Reference encoder stride
	ref_enc_pad = 1								# Reference encoder pad
	ref_enc_gru_dim = style_embedding_dim // 2  # Reference encoder GRU dim


	### Waveglow params
	n_flows = 12								# Number of steps of flow
	n_group = 8									# Number of samples in a group processed by the steps of flow
	n_early_every = 4							# Determines how often (i.e., after how many coupling layers) a number of channels (defined by --early-size parameter) are output to the loss function
	n_early_size = 2							# Number of channels output to the loss function
	wg_sigma = 1.0								# Standard deviation used for sampling from Gaussian
	segment_length = 4000						# Segment length (audio samples) processed per iteration
	wn_config = dict(
		n_layers=8,								# Number of layers in WN
		kernel_size=3,							# Kernel size for dialted convolution in the affine coupling layer (WN)
		n_channels=512							# Number of channels in WN
	)


	### LM Parameters
	use_lm = False
	language_model = 'gpt-2'
