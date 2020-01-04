# Tacotron 2 (multispeaker + gst) And WaveGlow

This Repository contains a sample code for Tacotron 2, WaveGlow with multi-speaker, emotion embedding together with a script for data preprocessing.  
Checkpoints and code originate from following sources:

* [Nvidia Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* [Nvidia Tacotron 2](https://github.com/NVIDIA/tacotron2)
* [Nvidia WaveGlow](https://github.com/NVIDIA/waveglow)
* [Torch Hub WaveGlow](https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/)
* [Torch Hub Tacotron 2](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/)


We've cleaned the code and changed it's structure, added multi-speaker and emotion embendings, added preprocessing, added restore / checkpoint mechanism and tensorboard. 


## Table of Contents
* [Setup](#setup)
   * [Requirements](#requirements)
* [Repository Description](#repository-description)
* [Data Preprocessing](#data-preprocessing)
* [Training](#training)
* [Inference process](#inference-process)
* [Parameters](#parameters)
     * [Shared parameters](#shared-parameters)
     * [Shared audio/STFT parameters](#shared-audiostft-parameters)
     * [Tacotron 2 parameters](#Tacotron-2-parameters)
     * [WaveGlow parameters](#waveglow-parameters)
* [TODOs](#todos)
   

## Setup

The following section lists the requirements in order to start training the
Tacotron 2 and WaveGlow models.


Clone the repository:
   ```bash
   git clone https://github.com/ide8/tacotron2  
   cd tacotron2
   PROJDIR=$(pwd)
   export PYTHONPATH=$PROJDIR:$PYTHONPATH
   ```

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container
and encapsulates some dependencies. Aside from these dependencies, ensure you
have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.06-py3+ NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
or newer
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU


Build an image from Docker file:

```bash
docker build --tag taco .
```
Run docker container:  
```bash
docker run --shm-size=8G --runtime=nvidia -v /absolute/path/to/your:/app -v /absolute/path/to/your:/data -v /absolute/path/to/your:/logs -v /absolute/path/to/your:/raw-data -detach taco sleep inf
```
Check container id:
```bash
docker ps
```
Select container id of image with tag `taco` and log into container with:
```
docker exec -it container_id bash
```

## Repository description

Folders `tacotron2` and `waveglow` have scripts for Tacotron 2, WaveGlow models and consists of:  

* `<model_name>/model.py` - model architecture
* `<model_name>/data_function.py` - data loading functions
* `<model_name>/loss_function.py` - loss function

Folder `common` contains common layers for both models (`common/layers.py`), utils (`common/utils.py`) and audio processing (`common/audio_processing.py` and `common/stft.py`).  

`router` folder is used by training script to select an appropriate model

In the root directory:
* `train.py` - script for model training
* `preprocess.py` - performs audio processing and creates training and validation datasets
* `inference.ipynb` - notebook for running inference

Folder `configs` contains `__init__.py` with all parameters needed for training and data processing. Folder `configs/experiments` consists of all the experiments. `default.py` is provided as an example.
On training or data processing start, parameters are copied from your experiment (in our case - `default.py`) to `__init__.py`, from which they are used by the system.


## Data preprocessing

First of all you need to preprocess your data. 
For each speaker you have to have a folder named with speaker name, containing wavs folder and metadata.csv file with the next line format: `file_name.wav|text`

All necessary parameters for preprocessiong should be set in `configs/experiments/default.py` in the class `PreprocessingConfig`.  


If  you're running preprocessing first time, set `start_from_preprocessed` flag to **False**, `preprocess.py` will perform trimming of audio files up to `PreprocessingConfig.top_db` parameter(cuts the silence in the beginning and end), applies ffmpeg command in order to mono, make same sampling rate and bit rate for all the wavs in dataset. 

It saves a folder `wavs` with processed audio files and `data.csv` file in `PreprocessingConfig.output_directory` with the following format: `path, text, speaker_name, speaker_id, emotion, text_len, duration`.  
Trimming and ffmpeg command are applied only to speakers, for which flag `process_audio` is **True**. Speakers with flag `emotion_present` is **False**, are treated as with emotion `neutral-normal`.

You won't need `start_from_preprocessed = False` once you finish running preprocessing script.
Only exception in case of new raw data comes in.
Once `start_from_preprocessed` is set to **True**, script will load file `data.csv` (from the `start_from_preprocessed = False` run), and forms `train.txt` and `val.txt` out from `data.csv`

`PreprocessingConfig.cpus` - defines number of cores for batch generator.
`PreprocessingConfig.sr` - defines sample ratio for reading and writing audio.
`PreprocessingConfig.emo_id_map` - dictionary for emotion mapping.
`PreprocessingConfig.data['path']` - is path to folder named with speaker name and containing `wavs` folder and `metadata.csv` with the following line format: `file_name.wav|text|emotion (optional)`.

Preprocessing script forms training and validation datasets in the following way:
* selects rows with audio duration and text length less or equal those for speaker `PreprocessingConfig.limit_by` (this is needed for proper batch size).
If such speaker is not present, than it selects rows within `PreprocessingConfig.text_limit` and `PreprocessingConfig.dur_limit`. Lower limit for audio is defined by `PreprocessingConfig.minimum_viable_dur`. For being able to use the same batch size as NVIDIA guys, set `PreprocessingConfig.text_limit` to `linda_jonson`.
* splits dataset randomly by ratio `train : valid = 0.95 : 0.05`
* if speaker train set is bigger than `PreprocessingConfig.n` - samples `n` rows
* saves `train.txt` and `val.txt` to `PreprocessingConfig.output_directory` 
* saves `emotion_coefficients.json` and `speaker_coefficients.json` with coefficients for loss balancing (used by `train.py`).  

Run preprocessing with:

```
python preprocess.py --exp default
```

## Training 
In `configs/experiment/default.py`, in the class `Config` set:
* `training_files` and `validation_files` - paths to `train.txt`, `val.txt`
* `tacotron_checkpoint`, `waveglow_checkpoint` - paths to pretrained Tacotron2 and Waveglow if they exist (we were able to restore Waveglow from Nvidia, but Tacotron2 code was edited to add speakers and emotions, so Tacotron2 needs to be trained from scratch)
* `output_directory` - path for writing checkpoints  


**Tacotron2**:

* Set `Config.model_name` to `"Tacotron2"`  
* Launch training

  * Single gpu: 
  ```
  python train.py --exp default
  ```
  * Multigpu training:  
  ```
  python -m multiproc train.py --exp default
  ```
  

**WaveGlow**:

* Set `Config.model_name` to `"WaveGlow"`  
* Launch training

  * Single gpu: 
  ```
  python train.py --exp default
  ```
  * Multigpu training:  
  ```
  python -m multiproc train.py --exp default
  ```

For restoring from checkpoint set path to checkpoints in `Config.restore_from`.  
In order to use emotions, set `use_emotions` as **True**.  
For balancing loss, set `use_loss_coefficients` as **True** and paths to jsons with coefficients in `emotion_coefficients` and `speaker_coefficients`.  


**Running Tensorboard**

Run `taco` Docker container. In Terminal execute:  
```
docker ps
```
Select Container ID of image with tag `taco` and run:

```
docker exec -it container_id bash
```

Start Tensorboard:  

```
 tensorboard --logdir=path_to_folder_with_logs --host=0.0.0.0
```

Loss together is being written into tensorboard:

![Tensorboard Scalars](/img/tacotron-scalars.png)

Audio samples are saved into tensorbaord as well each `Config.epochs_per_checkpoint`. Transcripts for audios are listed in `default.py/Config.phrases`

![Tensorboard Audio](/img/tacotron-audio.png)


## Inference process

Running inference with the `inference.ipynb` notebook.  

Run Jupyter Notebook:  
```
jupyter notebook --ip 0.0.0.0 --port 6006 --no-browser --allow-root
```

output:  
```
root@04096a19c266:/app# jupyter notebook --ip 0.0.0.0 --port 6006 --no-browser --allow-root
[I 09:31:25.393 NotebookApp] JupyterLab extension loaded from /opt/conda/lib/python3.6/site-packages/jupyterlab
[I 09:31:25.393 NotebookApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[I 09:31:25.395 NotebookApp] Serving notebooks from local directory: /app
[I 09:31:25.395 NotebookApp] The Jupyter Notebook is running at:
[I 09:31:25.395 NotebookApp] http://(04096a19c266 or 127.0.0.1):6006/?token=bbd413aef225c1394be3b9de144242075e651bea937eecce
[I 09:31:25.395 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 09:31:25.398 NotebookApp] 
    
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-15398-open.html
    Or copy and paste one of these URLs:
        http://(04096a19c266 or 127.0.0.1):6006/?token=bbd413aef225c1394be3b9de144242075e651bea937eecce
```

Select adress with 127.0.01 and put it in the browser.
In this case:
`http://127.0.0.1:6006/?token=bbd413aef225c1394be3b9de144242075e651bea937eecce`  

This script takes
text as input and runs Tacotron 2 and then WaveGlow inference to produce an
audio file. It requires  pre-trained checkpoints from Tacotron 2 and WaveGlow
models, input text, speaker_id and emotion_id.  

Change paths to checkpoints of pretrained WaveGlow and Tacotron2 in the cell [2] of the `inference.ipynb`.  
Write a text to be displayed in the cell [7] of the `inference.ipynb`.  


## Parameters

In this section, we list the most important hyperparameters,
together with their default values that are used to train Tacotron2 and
WaveGlow models.

### Shared parameters

* `epochs` - number of epochs (Tacotron 2: 1501, WaveGlow: 1001)
* `learning-rate` - learning rate (Tacotron 2: 1e-3, WaveGlow: 1e-4)
* `batch-size` - batch size (Tacotron 2 64, WaveGlow: 4)

### Shared audio/STFT parameters

* `sampling-rate` - sampling rate in Hz of input and output audio (22050)
* `filter-length` - (1024)
* `hop-length` - hop length for FFT, i.e., sample stride between consecutive FFTs (256)
* `win-length` - window size for FFT (1024)
* `mel-fmin` - lowest frequency in Hz (0.0)
* `mel-fmax` - highest frequency in Hz (8.000)

### Tacotron 2 parameters

* `anneal-steps` - epochs at which to anneal the learning rate (500/ 1000/ 1500)
* `anneal-factor` - factor by which to anneal the learning rate (0.1)

### WaveGlow parameters

* `segment-length` - segment length of input audio processed by the neural network (8000)


# TODOs

- [ ] Make Decoder work with n > 1 frames per step
- [ ] Make training work at FP16. 
