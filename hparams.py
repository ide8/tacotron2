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
	gst_use = False                              # Use or not to use GST
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


class PreprocessingConfig:
	code_path = '/workspace/code/gst'
	OUTPUT_DIRECTORY = '/workspace/training_data'
	SR = 22050                                   # sampling ratio for audio processing
	TOP_DB = 40                                  # level to trim audio
	limit_by = 'linda_johnson'                   # speaker to measure text_limit, dur_limit
	minimum_viable_dur = 0.05                    # min duration of audio
	text_limit = None                            # max text length (used by default)
	dur_limit = None                             # max audio duration (used by default)
	N = 100000                                   # max size of training dataset
	output_directory = '/media/olga/b40f5f55-fcbc-4f51-a740-22ed42f6902c/Olga/tacotron2/proc'
	data = [  # TODO: PEP8
		{	 # TODO: Dicts instead to tuples
			'path': '/media/olga/b40f5f55-fcbc-4f51-a740-22ed42f6902c/Olga/tacotron2/linda_johnson',
			'speaker_id': 0,
			'process_audio': False
		},
		{
			'path': '/media/olga/b40f5f55-fcbc-4f51-a740-22ed42f6902c/Olga/tacotron2/scarjo_the_dive_descript_grouped_50mil',
			'speaker_id': 1,
			'process_audio': True
		},
		{
			'path': '/media/olga/b40f5f55-fcbc-4f51-a740-22ed42f6902c/Olga/tacotron2/scarjo_the_dive_descript_ungrouped',
			'speaker_id': 1,
			'process_audio': True
		}
		# {
		#     'path':'/workspace/data/gcp/samantha_default',
		#     'speaker_id':1,
		#     'process_audio':True
		# },
		# {
		#     'path':'/workspace/data/scarjo_her',
		#     'speaker_id':1,
		#     'process_audio':True,
		# },
		# {
		#     'path':'/workspace/data/aws/dataset/blizzard_2013',
		#     'speaker_id':2,
		#     'process_audio':True
		# },
		# {
		#     'path':'/workspace/data/aws/dataset/en_US/by_book/female/judy_bieber',
		#     'speaker_id':3,
		#     'process_audio':True
		# },
		# {
		#     'path':'/workspace/data/aws/dataset/en_US/by_book/female/mary_ann',
		#     'speaker_id':4,
		#     'process_audio':True
		# },
		# {
		#     'path':'/workspace/data/aws/dataset/en_UK/by_book/female/elizabeth_klett',
		#     'speaker_id':5,
		#     'process_audio':True
		# },
		# {
		#     'path':'/workspace/data/aws/dataset/en_US/by_book/male/elliot_miller',
		#     'speaker_id':6,
		#     'process_audio':True
		# }
	]