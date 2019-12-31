from tacotron2.text import symbols
global symbols


class Config:
    # Audio params
    sampling_rate = 22050                        # Sampling rate
    filter_length = 1024                         # Filter length
    hop_length = 256                             # Hop (stride) length
    win_length = 1024                            # Window length
    mel_fmin = 0.0                               # Minimum mel frequency
    mel_fmax = 8000.0                            # Maximum mel frequency
    n_mel_channels = 80                          # Number of bins in mel-spectrograms
    max_wav_value = 32768.0                      # Maximum audiowave value

    ### Tacotron Params
    # Optimization
    mask_padding = False                         # Use mask padding

    # Symbols
    n_symbols = len(symbols)                     # Number of symbols in dictionary
    symbols_embedding_dim = 512                  # Text input embedding dimension

    # Speakers
    n_speakers = 128                             # Number of speakers
    speakers_embedding_dim = 16                  # Speaker embedding dimension

    # Emotions
    use_emotions = True                         # Use emotions
    n_emotions = 15                             # N emotions
    emotions_embedding_dim = 8                  # Emotion embedding dimension
    balance_loss_emotions = True                # to balance loss by emotions class ratio
    balance_loss_speakers = True                # to balance loss accordingly by speaker class ratio
    coef_file_emotions = '/train/coefficients_emotions.json'       # path to dict with emotions coefficients
    coef_file_speakers = '/train/coefficients_speakers.json'       # path to dict with speaker coefficients

    # Encoder
    encoder_kernel_size = 5                      # Encoder kernel size
    encoder_n_convolutions = 3                   # Number of encoder convolutions
    encoder_embedding_dim = 512                  # Encoder embedding dimension

    # Attention
    attention_rnn_dim = 1024                     # Number of units in attention LSTM
    attention_dim = 128                          # Dimension of attention hidden representation

    # Attention location
    attention_location_n_filters = 32            # Number of filters for location-sensitive attention
    attention_location_kernel_size = 31          # Kernel size for location-sensitive attention

    # Decoder
    n_frames_per_step = 1                        # Number of frames processed per step
    decoder_rnn_dim = 1024                       # Number of units in decoder LSTM
    prenet_dim = 256                             # Number of ReLU units in prenet layers
    max_decoder_steps = 2000                     # Maximum number of output mel spectrograms
    gate_threshold = 0.5                         # Probability threshold for stop token
    p_attention_dropout = 0.1                    # Dropout probability for attention LSTM
    p_decoder_dropout = 0.1                      # Dropout probability for decoder LSTM
    decoder_no_early_stopping = False            # Stop decoding once all samples are finished

    # Postnet
    postnet_embedding_dim = 512                  # Postnet embedding dimension
    postnet_kernel_size = 5                      # Postnet kernel size
    postnet_n_convolutions = 5                   # Number of postnet convolutions

    ### Waveglow params
    n_flows = 12                                  # Number of steps of flow
    n_group = 8                                   # Number of samples in a group processed by the steps of flow
    n_early_every = 4                             # Determines how often (i.e., after how many coupling layers) a number of channels (defined by --early-size parameter) are output to the loss function
    n_early_size = 2                              # Number of channels output to the loss function
    wg_sigma = 1.0                                # Standard deviation used for sampling from Gaussian
    segment_length = 4000                         # Segment length (audio samples) processed per iteration
    wn_config = dict(
        n_layers=8,                               # Number of layers in WN
        kernel_size=3,                            # Kernel size for dialted convolution in the affine coupling layer (WN)
        n_channels=512                            # Number of channels in WN
    )

    ### Script args
    model_name = "Tacotron2"
    output_directory = "/logs"                                                                                          # Directory to save checkpoints
    log_file = "nvlog.json"                                                                                             # Filename for logging

    anneal_steps = [500, 1000, 1500]                                                                                    # Epochs after which decrease learning rate
    anneal_factor = 0.1                                                                                                 # Factor for annealing learning rate

    tacotron2_checkpoint = '/data/pretrained/t2_fp32_torch'   # Path to pre-trained Tacotron2 checkpoint for sample generation
    waveglow_checkpoint = '/data/pretrained/wg_fp32_torch'    # Path to pre-trained WaveGlow checkpoint for sample generation
    restore_from = ''                                                       # Checkpoint path to restore from

    # Training params
    epochs = 1910                                             # Number of total epochs to run
    epochs_per_checkpoint = 20                                # Number of epochs per checkpoint
    seed = 1234                                               # Seed for PyTorch random number generators
    dynamic_loss_scaling = True                               # Enable dynamic loss scaling
    amp_run = False                                           # Enable AMP
    cudnn_enabled = True                                      # Enable cudnn
    cudnn_benchmark = False                                   # Run cudnn benchmark

    # Optimization params
    use_saved_learning_rate = False
    learning_rate = 1e-3                                      # Learning rate
    weight_decay = 1e-6                                       # Weight decay
    grad_clip_thresh = 1.0                                    # Clip threshold for gradients
    batch_size = 40                                           # Batch size per GPU
    grad_clip = 5.0                                           # Enables gradient clipping and sets maximum gradient norm value

    # Dataset
    load_mel_from_dist = False                                # Loads mel spectrograms from disk instead of computing them on the fly
    text_cleaners = ['english_cleaners']                      # Type of text cleaners for input text
    #training_files = '/data/train.txt'                       # Path to training filelist
    #validation_files = '/data/val.txt'                       # Path to validation filelist
    training_files = '/train/train.txt'
    validation_files = '/train/val.txt'


    dist_url = 'tcp://localhost:23456'                        # Url used to set up distributed training
    group_name = "group_name"                                 # Distributed group name
    dist_backend = "nccl"                                     # Distributed run backend

    # Sample phrases
    phrases = {
        'speaker_ids': [0, 1],
        'texts': [
            'Hello, how are you doing today?',
            'I would like to eat a Hamburger.',
            'Hi.',
            'I would like to eat a Hamburger. Would you like to join me?',
            'Do you have any hobbies?'
        ]
    }


class PreprocessingConfig:
    cpus = 42                                    # Amount of cpus for parallelization
    sr = 22050                                   # sampling ratio for audio processing
    top_db = 40                                  # level to trim audio
    limit_by = 'linda_johnson'                   # speaker to measure text_limit, dur_limit
    minimum_viable_dur = 0.05                    # min duration of audio
    text_limit = 188                            # max text length (used by default)
    dur_limit = 10                             # max audio duration (used by default)
    n = 100000                                   # max size of training dataset per speaker
    start_from_preprocessed = False              # load data.csv - should be in output_directory
    save_balance_emotions = True
    save_balance_speaker = True
    #output_directory = '/data'
    output_directory = '/train'
    data = [
        {
             'path': '/data/raw-data/linda_johnson',
            'speaker_id': 0,
             'process_audio': False,
             'emotion_present': False
         },
        # {
        #    'path': '/raw-data/scarjo_the_dive_descript_grouped_50mil',
        #    'speaker_id': 1,
        #    'process_audio': True,
        #    'emotion_present': False
        # },
        # {
        #    'path': '/raw-data/scarjo_the_dive_descript_ungrouped',
        #    'speaker_id': 1,
        #    'process_audio': True,
        #    'emotion_present': False
        # },
        {
            'path': '/data/raw-data/mellisa',
            'speaker_id': 1,
            'process_audio': True,
            'emotion_present': True
        }
    ]

    emo_id_map = {
        'neutral-normal': 0,
        'calm-normal': 1,
        'calm-strong': 2,
        'happy-normal': 3,
        'happy-strong': 4,
        'sad-normal': 5,
        'sad-strong': 6,
        'angry-normal': 7,
        'angry-strong': 8,
        'fearful-normal': 9,
        'fearful-strong': 10,
        'disgust-normal': 11,
        'disgust-strong': 12,
        'surprised-normal': 13,
        'surprised-strong': 14
    }
