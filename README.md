# Tacotron 2 (multispeaker + gst) And WaveGlow

This repository provides a script and recipe to train Tacotron 2 and WaveGlow
v1.6 models to achieve state of the art accuracy, and is tested and maintained by NVIDIA.

## Table of Contents
* [Setup](#setup)
   * [Requirements](#requirements)
* [Quick Start Guide](#quick-start-guide)
* [Advanced](#advanced)
   * [Scripts and sample code](#scripts-and-sample-code)
   * [Parameters](#parameters)
      * [Shared parameters](#shared-parameters)
      * [Shared audio/STFT parameters](#shared-audiostft-parameters)
      * [Tacotron 2 parameters](#tacotron-2-parameters)
      * [WaveGlow parameters](#waveglow-parameters)
   * [Command-line options](#command-line-options)
   * [Getting the data](#getting-the-data)
      * [Dataset guidelines](#dataset-guidelines)
      * [Multi-dataset](#multi-dataset)
   * [Training process](#training-process)
   * [Inference process](#inference-process)
* [Performance](#performance)
   * [Benchmarking](#benchmarking)
      * [Training performance benchmark](#training-performance-benchmark)
      * [Inference performance benchmark](#inference-performance-benchmark)
   * [Results](#results)
      * [Training accuracy results](#training-accuracy-results)
         * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-8x-v100-16g)
      * [Training performance results](#training-performance-results)
         * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-8x-v100-16g)
         * [Expected training time](#expected-training-time)
      * [Inference performance results](#inference-performance-results)
         * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-8x-v100-16g)
* [Release notes](#release-notes)
   * [Changelog](#changelog)
   * [Known issues](#known-issues)

## Setup

The following section lists the requirements in order to start training the
Tacotron 2 and WaveGlow models.

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container
and encapsulates some dependencies. Aside from these dependencies, ensure you
have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.06-py3+ NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
or newer
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning
Documentation:

* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
* [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
* [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html#running)

For those unable to use the PyTorch NGC container, to set up the required
environment or create your own container, see the versioned
[NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed precision with Tensor Cores or using FP32,
perform the following steps using the default parameters of the Tacrotron 2
and WaveGlow model on the [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
dataset.

1. Clone the repository.
   ```bash
   git clone https://github.com/ide8/tacotron2_waveglow_multispeaker_gst
   cd tacotron2_waveglow_multispeaker_gst
   PROJDIR=$(pwd)
   export PYTHONPATH=$PROJDIR:$PYTHONPATH
   ```

2. For each speaker you have to have a folder named with speaker name, containing `wavs` folder and `metadata.csv` file with the next line format: `full_path_to_wav|text`.

3. After that you run this script. It requires ffmpeg to be installed. It’s pretty intense script and it uses Pool to parallelize jobs. Now there is 42 (I used it on compute instance with 64 cores where 22 of them where busy, that’s why there is 42 and not because it’s an answer on general question :) ), so you put the number of cores you have available. After running this script you can have a coffee or some walk in the garden. It will create a folder for each speaker named with speaker name under `output_path`.

4. Once `preprocess.py` finishes it’s job you need to go to the `check_distributions.py` under the same folder. There are two variables to change: `speakers` and `data_path` (equals to `output_path` in previous file). Speakers it’s basically an array of speaker names which correspond to folder names. This file is very important in terms of two things, if you want to have the same batch size as NVIDIA guys, you need to have all the files match the distribution of the LJ dataset which NVIDIA guys used. Having that `data.pickle` (an output of `check_distributions.py` file) you can check the distributions of your data with pandas for example. You need to check the symbols len (after they are tokenized) and duration - as I remember it must not exceed ~180 symbols and ~10sec. In case you don’t want to do that - just include LJ dataset as well. After that run check_distributions file, it is also resource intense, and there is also 42 for parallelism. 

5. Once you finish with that, go to the last file `form_train_val_set.py`, it is for sampling, it actually uses linda_jonson (LJ) to cut larger and empty files of the other speakers. Also there is another hard coded thing goes to `samantha` speaker, I have very small amount of wavs for this speaker that’s why I simply added each wav two times. Also there is a maximum amount of wavs for each speaker under N variable and dict with mapping speaker name to its id. Once you adjust it you can run it - it’s pretty fast. This file finally generates `train.txt` and `val.txt` files.


*flag - this is needed for almost all the datasets except a few very good, once it set to true it will cut the silence in the beginning and the end of wav and also convert it to 22KHz and 16 bit rate, which is essential.

6. Set `os.environ["CUDA_VISIBLE_DEVICES"]` to GPUs you want you want to use in multigpu setting.

7. Launch multigpu training:
`python -m multiproc train.py -m Tacotron2 -o {output_path} -lr 1e-3 --epochs 10 -bs 64 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file {output_path/nvlog.json} --training-files {training_data_path/train.txt} --validation-files {val_data_path/val.txt} --anneal-steps 500 1000 1500 --anneal-factor 0.1`

8. To restore from checkpoint you just add `--restore-from {checkpoint_path}` to the training command


## Advanced

The following sections provide greater details of the dataset, running
training and inference, and the training results.

### Scripts and sample code

The sample code for Tacotron 2 and WaveGlow has scripts specific to a
particular model, located in directories `./tacotron2` and `./waveglow`, as well as scripts common to both
models, located in the `./common` directory. The model-specific scripts are as follows:

* `<model_name>/model.py` - the model architecture, definition of forward and
inference functions
* `<model_name>/arg_parser.py` - argument parser for parameters specific to a
given model
* `<model_name>/data_function.py` - data loading functions
* `<model_name>/loss_function.py` - loss function for the model

The common scripts contain layer definitions common to both models
(`common/layers.py`), some utility scripts (`common/utils.py`) and scripts
for audio processing (`common/audio_processing.py` and `common/stft.py`). In
the root directory `./` of this repository, the `./run.py` script is used for
training while inference can be executed with the `./inference.py` script. The
scripts `./models.py`, `./data_functions.py` and `./loss_functions.py` call
the respective scripts in the `<model_name>` directory, depending on what
model is trained using the `run.py` script.

### Parameters

In this section, we list the most important hyperparameters and command-line arguments,
together with their default values that are used to train Tacotron 2 and
WaveGlow models.

#### Shared parameters

* `--epochs` - number of epochs (Tacotron 2: 1501, WaveGlow: 1001)
* `--learning-rate` - learning rate (Tacotron 2: 1e-3, WaveGlow: 1e-4)
* `--batch-size` - batch size (Tacotron 2 FP16/FP32: 128/64, WaveGlow FP16/FP32: 10/4)
* `--amp-run` - use mixed precision training

#### Shared audio/STFT parameters

* `--sampling-rate` - sampling rate in Hz of input and output audio (22050)
* `--filter-length` - (1024)
* `--hop-length` - hop length for FFT, i.e., sample stride between consecutive FFTs (256)
* `--win-length` - window size for FFT (1024)
* `--mel-fmin` - lowest frequency in Hz (0.0)
* `--mel-fmax` - highest frequency in Hz (8.000)

#### Tacotron 2 parameters

* `--anneal-steps` - epochs at which to anneal the learning rate (500 1000 1500)
* `--anneal-factor` - factor by which to anneal the learning rate (FP16/FP32: 0.3/0.1)

#### WaveGlow parameters

* `--segment-length` - segment length of input audio processed by the neural network (8000)


### Command-line options

To see the full list of available options and their descriptions, use the `-h` 
or `--help` command line option, for example:
```bash
python train.py --help
```

The following example output is printed when running the sample:

```bash
Batch: 7/260 epoch 0
:::NVLOGv0.2.2 Tacotron2_PyT 1560936205.667271376 (/workspace/tacotron2/dllogger/logger.py:251) train_iter_start: 7
:::NVLOGv0.2.2 Tacotron2_PyT 1560936207.209611416 (/workspace/tacotron2/dllogger/logger.py:251) train_iteration_loss: 5.415428161621094
:::NVLOGv0.2.2 Tacotron2_PyT 1560936208.705905914 (/workspace/tacotron2/dllogger/logger.py:251) train_iter_stop: 7
:::NVLOGv0.2.2 Tacotron2_PyT 1560936208.706479311 (/workspace/tacotron2/dllogger/logger.py:251) train_iter_items/sec: 8924.00136085362
:::NVLOGv0.2.2 Tacotron2_PyT 1560936208.706998110 (/workspace/tacotron2/dllogger/logger.py:251) iter_time: 3.0393316745758057
Batch: 8/260 epoch 0
:::NVLOGv0.2.2 Tacotron2_PyT 1560936208.711485624 (/workspace/tacotron2/dllogger/logger.py:251) train_iter_start: 8
:::NVLOGv0.2.2 Tacotron2_PyT 1560936210.236668825 (/workspace/tacotron2/dllogger/logger.py:251) train_iteration_loss: 5.516331672668457
```


### Getting the data

The Tacotron 2 and WaveGlow models were trained on the LJSpeech-1.1 dataset.  
This repository contains the `./scripts/prepare_dataset.sh` script which will automatically download and extract the whole dataset. By default, data will be extracted to the `./LJSpeech-1.1` directory. The dataset directory contains a `README` file, a `wavs` directory with all audio samples, and a file `metadata.csv` that contains audio file names and the corresponding transcripts.

#### Dataset guidelines

The LJSpeech dataset has 13,100 clips that amount to about 24 hours of speech. Since the original dataset has all transcripts in the `metadata.csv` file, in this repository we provide file lists in the `./filelists` directory that determine training and validation subsets; `ljs_audio_text_train_filelist.txt` is a test set used as a training dataset and `ljs_audio_text_val_filelist.txt` is a test set used as a validation dataset.

#### Multi-dataset

To use datasets different than the default LJSpeech dataset:

1. Prepare a directory with all audio files and pass it to the `--dataset-path` command-line option.  

2. Add two text files containing file lists: one for the training subset (`--training-files`) and one for the validation subset (`--validation files`).
The structure of the filelists should be as follows:
   ```bash
   `<audio file path>|<transcript>`
   ```

   The `<audio file path>` is the relative path to the path provided by the `--dataset-path` option.

### Training process

The Tacotron2 and WaveGlow models are trained separately and independently.
Both models obtain mel-spectrograms from short time Fourier transform (STFT)
during training. These mel-spectrograms are used for loss computation in case
of Tacotron 2 and as conditioning input to the network in case of WaveGlow.

The training loss is averaged over an entire training epoch, whereas the
validation loss is averaged over the validation dataset. Performance is
reported in total output mel-spectrograms per second for the Tacotron 2 model and
in total output samples per second for the WaveGlow model. Both measures are
recorded as `train_iter_items/sec` (after each iteration) and
`train_epoch_items/sec` (averaged over epoch) in the output log file `./output/nvlog.json`. The result is
averaged over an entire training epoch and summed over all GPUs that were
included in the training.

Even though the training script uses all available GPUs, you can change
this behavior by setting the `CUDA_VISIBLE_DEVICES` variable in your
environment or by setting the `NV_GPU` variable at the Docker container launch
([see section "GPU isolation"](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#gpu-isolation)).

### Inference process

You can run inference using the `./inference.py` script. This script takes
text as input and runs Tacotron 2 and then WaveGlow inference to produce an
audio file. It requires  pre-trained checkpoints from Tacotron 2 and WaveGlow
models and input text as a text file, with one phrase per line.

To run inference, issue:
```bash
python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> -o output/ --include-warmup -i phrases/phrase.txt --amp-run
```
Here, `Tacotron2_checkpoint` and `WaveGlow_checkpoint` are pre-trained
checkpoints for the respective models, and `phrases/phrase.txt` contains input phrases. The number of text lines determines the inference batch size. Audio will be saved in the output folder.

You can find all the available options by calling `python inference.py --help`.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model
performance in training and inference mode.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

**Tacotron 2**

* For 1 GPU
	* FP16
        ```bash
        python train.py -m Tacotron2 -o <output_dir> -lr 1e-3 --epochs 10 -bs <batch_size> --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_2500_filelist.txt --dataset-path <dataset-path> --amp-run
        ```
	* FP32
        ```bash
        python train.py -m Tacotron2 -o <output_dir> -lr 1e-3 --epochs 10 -bs <batch_size> --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_2500_filelist.txt --dataset-path <dataset-path>
        ```

* For multiple GPUs
	* FP16
        ```bash
        python -m multiproc train.py -m Tacotron2 -o <output_dir> -lr 1e-3 --epochs 10 -bs <batch_size> --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_2500_filelist.txt --dataset-path <dataset-path> --amp-run
        ```
	* FP32
        ```bash
        python -m multiproc train.py -m Tacotron2 -o <output_dir> -lr 1e-3 --epochs 10 -bs <batch_size> --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_2500_filelist.txt --dataset-path <dataset-path>
        ```

**WaveGlow**

* For 1 GPU
	* FP16
        ```bash
        python train.py -m WaveGlow -o <output_dir> -lr 1e-4 --epochs 10 -bs <batch_size> --segment-length 8000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt --dataset-path <dataset-path> --amp-run
        ```
	* FP32
        ```bash
        python train.py -m WaveGlow -o <output_dir> -lr 1e-4 --epochs 10 -bs <batch_size> --segment-length  8000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt --dataset-path <dataset-path>
        ```

* For multiple GPUs
	* FP16
        ```bash
        python -m multiproc train.py -m WaveGlow -o <output_dir> -lr 1e-4 --epochs 10 -bs <batch_size> --segment-length 8000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt --dataset-path <dataset-path> --amp-run
        ```
	* FP32
        ```bash
        python -m multiproc train.py -m WaveGlow -o <output_dir> -lr 1e-4 --epochs 10 -bs <batch_size> --segment-length 8000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt --dataset-path <dataset-path>
        ```

Each of these scripts runs for 10 epochs and for each epoch measures the
average number of items per second. The performance results can be read from
the `nvlog.json` files produced by the commands.

#### Inference performance benchmark

To benchmark the inference performance on a batch size=1, run:

* For FP16
    ```bash
    python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> -o output/ --include-warmup -i phrases/phrase_1_64.txt --amp-run --log-file=output/nvlog_fp16.json
    ```
* For FP32
    ```bash
    python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> -o output/ --include-warmup -i phrases/phrase_1_64.txt --log-file=output/nvlog_fp32.json
    ```

The output log files will contain performance numbers for Tacotron 2 model
(number of output mel-spectrograms per second, reported as `tacotron2_items_per_sec`) 
and for WaveGlow (number of output samples per second, reported as `waveglow_items_per_sec`). 
The `inference.py` script will run a few warmup iterations before running the benchmark. 

### Results

The following sections provide details on how we achieved our performance
and accuracy in training and inference.

#### Training accuracy results

##### NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `./platform/train_{tacotron2,waveglow}_{AMP,FP32}_DGX1_16GB_8GPU.sh` training script in the PyTorch-19.06-py3
NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs.

All of the results were produced using the `train.py` script as described in the
[Training process](#training-process) section of this document.

| Loss (Model/Epoch) |       1 |     250 |     500 |     750 |    1000 |
| :----------------: | ------: | ------: | ------: | ------: | ------: |
| Tacotron 2 FP16 | 13.0732 | 0.5736 | 0.4408 | 0.3923 | 0.3735 |
| Tacotron 2 FP32 | 8.5776 | 0.4807 | 0.3875 | 0.3421 | 0.3308 |
| WaveGlow FP16  | -2.2054 | -5.7602 |  -5.901 | -5.9706 | -6.0258 |
| WaveGlow FP32  | -3.0327 |  -5.858 | -6.0056 | -6.0613 | -6.1087 |

Tacotron 2 FP16 loss - batch size 128 (mean and std over 16 runs)
![](./img/tacotron2_amp_loss.png "Tacotron 2 FP16 loss")

Tacotron 2 FP32 loss - batch size 64 (mean and std over 16 runs)
![](./img/tacotron2_fp32_loss.png "Tacotron 2 FP16 loss")

WaveGlow FP16 loss - batch size 10 (mean and std over 16 runs)
![](./img/waveglow_fp16_loss.png "WaveGlow FP16 loss")

WaveGlow FP32 loss - batch size 4 (mean and std over 16 runs)
![](./img/waveglow_fp32_loss.png "WaveGlow FP32 loss")


#### Training performance results

##### NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `./platform/train_{tacotron2,waveglow}_{AMP,FP32}_DGX1_16GB_8GPU.sh`
training script in the PyTorch-19.06-py3 NGC container on NVIDIA DGX-1 with
8x V100 16G GPUs. Performance numbers (in output mel-spectrograms per second for
Tacotron 2 and output samples per second for WaveGlow) were averaged over
an entire training epoch.

This table shows the results for Tacotron 2:

|Number of GPUs|Batch size per GPU|Number of mels used with mixed precision|Number of mels used with FP32|Speed-up with mixed precision|Multi-GPU weak scaling with mixed precision|Multi-GPU weak scaling with FP32|
|---:|---:|---:|---:|---:|---:|---:|
|1|128@FP16, 64@FP32 | 20,992  | 12,933 | 1.62 | 1.00 | 1.00 |
|4|128@FP16, 64@FP32 | 74,989  | 46,115 | 1.63 | 3.57 | 3.57 |
|8|128@FP16, 64@FP32 | 140,060 | 88,719 | 1.58 | 6.67 | 6.86 |

The following table shows the results for WaveGlow:

|Number of GPUs|Batch size per GPU|Number of samples used with mixed precision|Number of samples used with FP32|Speed-up with mixed precision|Multi-GPU weak scaling with mixed precision|Multi-GPU weak scaling with FP32|
|---:|---:|---:|---:|---:|---:|---:|
|1| 10@FP16, 4@FP32 | 81,503 | 36,671 | 2.22 | 1.00 | 1.00 |
|4| 10@FP16, 4@FP32 | 275,803 | 124,504 | 2.22 | 3.38 | 3.40 |
|8| 10@FP16, 4@FP32 | 583,887 | 264,903 | 2.20 | 7.16 | 7.22 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Expected training time

The following table shows the expected training time for convergence for Tacotron 2 (1501 epochs):

|Number of GPUs|Batch size per GPU|Time to train with mixed precision (Hrs)|Time to train with FP32 (Hrs)|Speed-up with mixed precision|
|---:|---:|---:|---:|---:|
|1| 128@FP16, 64@FP32 | 153 | 234 | 1.53 |
|4| 128@FP16, 64@FP32 | 42 | 64 | 1.54 |
|8| 128@FP16, 64@FP32 | 22 | 33 | 1.52 |

The following table shows the expected training time for convergence for WaveGlow (1001 epochs):

|Number of GPUs|Batch size per GPU|Time to train with mixed precision (Hrs)|Time to train with FP32 (Hrs)|Speed-up with mixed precision|
|---:|---:|---:|---:|---:|
|1| 10@FP16, 4@FP32 | 347 | 768 | 2.21 |
|4| 10@FP16, 4@FP32 | 106 | 231 | 2.18 |
|8| 10@FP16, 4@FP32 | 49 | 105 | 2.16 |

#### Inference performance results

##### NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `./inference.py` inference script in 
the PyTorch-19.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs.
Performance numbers (in output mel-spectrograms per second for Tacotron 2 and 
output samples per second for WaveGlow) were averaged over 16 runs.

The following table shows the inference performance results for Tacotron 2 model. 
Results are measured in the number of output mel-spectrograms per second.

|Number of GPUs|Number of mels used with mixed precision|Number of mels used with FP32|Speed-up with mixed precision|
|---:|---:|---:|---:|
|**1**|625|613|1.02|

The following table shows the inference performance results for WaveGlow model. 
Results are measured in the number of output samples per second<sup>1</sup>.

|Number of GPUs|Number of samples used with mixed precision|Number of samples used with FP32|Speed-up with mixed precision|
|---:|---:|---:|---:|
|**1**|180474|162282|1.11|

<sup>1</sup>With sampling rate equal to 22050, one second of audio is generated from 22050 samples.

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).


## Release notes

### Changelog

March 2019
* Initial release

June 2019
* AMP support
* Data preprocessing for Tacotron 2 training
* Fixed dropouts on LSTMCells

July 2019
* Changed measurement units for Tacotron 2 training and inference performance 
benchmarks from input tokes per second to output mel-spectrograms per second
* Introduced batched inference
* Included warmup in the inference script

### Known issues

There are no known issues in this release.
