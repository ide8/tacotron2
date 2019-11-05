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

    # GST
    gst_use = False                              # Use or not to use GST
    gst_n_tokens = 10                            # Number of GSTs
    gst_n_heads = 16                             # Number of heads in GST multi-head attention

    # Reference Encoder
    style_embedding_dim = 256                     # Style embedding dim
    ref_enc_filters = [32, 32, 64, 64, 128, 128]  # Reference encoder filter
    ref_enc_kernel_size = 3                       # Reference encoder kernel size
    ref_enc_stride = 2                            # Reference encoder stride
    ref_enc_pad = 1                               # Reference encoder pad
    ref_enc_gru_dim = style_embedding_dim // 2    # Reference encoder GRU dim


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

    ### LM Parameters
    use_lm = False
    language_model = 'gpt-2'

    ### Script args
    model_name = "Tacotron2"
    output_directory = "/workspace/output/"       # Directory to save checkpoints
    log_file = "nvlog.json"                       # Filename for logging

    # TODO: Add path phrases for validation and writing to TensorBoard
    phrase_path = None

    anneal_steps = None                           # Epochs after which decrease learning rate
    anneal_factor = 0.1                           # Factor for annealing learning rate

    tacotron2_checkpoint = \
        "/workspace/output/t2_fp32_torch"         # Path to pre-trained Tacotron2 checkpoint for sample generation
    waveglow_checkpoint = \
        "/workspace/output/wg_fp32_torch"         # Path to pre-trained WaveGlow checkpoint for sample generation

    restore_from = None                           # Checkpoint path to restore from
    tensorboard_log_dir = "/workspace/logs"       # TensorBoard logs save directory location.

    # Training params
    epochs = 10                                   # Number of total epochs to run
    epochs_per_checkpoint = 50                    # Number of epochs per checkpoint
    seed = 1234                                   # Seed for PyTorch random number generators
    dynamic_loss_scaling = True                   # Enable dynamic loss scaling
    amp_run = False                               # Enable AMP
    cudnn_enabled = True                          # Enable cudnn
    cudnn_benchmark = False                       # Run cudnn benchmark

    # Optimization params
    use_saved_learning_rate = False
    learning_rate = 0.001                         # Learing rate
    weight_decay = 0.00001                        # Weight decay
    grad_clip_thresh = 1.0                        # Clip threshold for gradients
    batch_size = 64                               # Batch size per GPU
    grad_clip = 5.0                               # Enables gradient clipping and sets maximum gradient norm value

    # Dataset
    load_mel_from_dist = False                    # Loads mel spectrograms from disk instead of computing them on the fly
    text_cleaners = ['english_cleaners']          # Type of text cleaners for input text
    training_files = \
        "/workspace/training_data/train.txt"      # Path to training filelist
    validation_files = \
        "/workspace/training_data/val.txt"        # Path to validation filelist

    # Distributed
    dist_url = 'tcp://localhost:23456'            # Url used to set up distributed training
    group_name = "group_name"                     # Distributed group name
    dist_backend = "nccl"                         # Distributed run backend

