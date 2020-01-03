# Tacotron 2 (multispeaker + gst) And WaveGlow

This Repository contains a sample code for Tacotron 2, WaveGlow with multi-speaker, emotion embedding together with a script for data preprocessing.  
Checkpoints and code originate from following sources:

* [Nvidia Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* [Nvidia Tacotron 2](https://github.com/NVIDIA/tacotron2)
* [Nvidia WaveGlow](https://github.com/NVIDIA/waveglow)
* [Pytorch WaveGlow](https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/)
* [Pytorch Tacotron 2](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/)


We add multi-speaker and emotion embending, changed preprocessing, add restore / checkpoint mechanism and tensorboard. 


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
docker build --tag taco
```
Run:  
```bash
docker run --shm-size=8G --runtime=nvidia -v /absolute/path/to/your:/app -v /absolute/path/to/your:/data -v /absolute/path/to/your:/logs -v /absolute/path/to/your:/raw-data -detach taco sleep inf
```

```bash
docker ps
```
Select Container ID of image with tag `taco` and run:

```
docker exec -it container_id bash
```

## Repository description


Folders `tacotron2` and `waveglow` have scripts for Tacotron 2, WaveGlow models and consists of:  

* `<model_name>/model.py` - the model architecture, definition of forward and
inference functions
* `<model_name>/data_function.py` - data loading functions
* `<model_name>/loss_function.py` - loss function for the model

Folder `common` contains common for both models layers (`common/layers.py`), utils (`common/utils.py`) and audio processing (`common/audio_processing.py` and `common/stft.py`).  
Scripts in `router` directory are used by training script to select an appropriate model.  

In the root directory:  
* `train.py` - script for model training
* `preprocess.py` - performs audio processing and make training and validation datasets
* `inference.ipynb` - notebook for running inference

Folder `Config` contains `__init__.py` with all parameters needed for training and data processing. Folder `Config/experiment` provides an example parameters in script `default.py`. 
When training or data processing run, parameters are copying from `default.py` to `__init__.py`


## Data preprocessing

 All necessary parameters for preprocessiong should be set in `configs/experiments/default.py` in the class `PreprocessingConfig`.  
 If `start_from_preprocessed` flag is set to **False**, `preprocess.py` performs trimming of audio files up to `PreprocessingConfig.top_db` parameter, applies ffmpeg command,
measures duration of audio files and lengths of correspondent text transcripts.  
 It saves a folder `wavs` with processed audio files and `data.csv` file in `PreprocessingConfig.output_directory` with the following format: `path_to_file, text, speaker_name, speaker_id, emotion, duration_of_audio, length_of_text`.  
Trimming and ffmpeg command are applied only to speakers, for which flag `process_audion` is **True**. Speakers with flag `emotion_present` is **False**, are treated as with emotion `neutral-normal`.  
If flag `start_from_preprocessed` is set to **True**, script loads file `data.csv` (it should be present in `PreprocessingConfig.output_directory`).  

`PreprocessingConfig.cpus` defines number of cores to use when parallelize jobs.  
`PreprocessingConfig.sr` defines sample ratio for reading and writing audio.  
`PreprocessingConfig.emo_id_map` - dictionary for emotion mapping.  
`PreprocessingConfig.data['path']` is path to folder named with speaker name and containing `wavs` folder and `metadata.csv` with the following line format: `full_path_to_wav|text|emotion`.

Preprocessing script forms training and validation datasets in the following way:
* selects rows with audio duration and text length less or equal those for speaker `PreprocessingConfig.limit_by`.  
If can't find, than it selects rows within `PreprocessingConfig.text_limit` and `PreprocessingConfig.dur_limit`. Lower limit for audio is defined by `PreprocessingConfig.minimum_viable_dur`. For being able to use the same batch size as NVIDIA guys, set `PreprocessingConfig.text_limit` to `linda_jonson`.
* splits dataset randomly by ratio `train : valid = 0.95 : 0.05`
* if train set is bigger than `PreprocessingConfig.n`, samples `n` rows
* saves `train.txt` and `val.txt` to `PreprocessingConfig.output_directory`  
It saves `emotion_coefficients.json` and `speaker_coefficients.json` with coefficients for loss balancing (used by `train.py`).  

Run preprocessing with:

```
python preprocess.py --exp default
```

## Training 
In `configs/experiment/default.py`, in the class `Config` set: 
*  `training_files` and `validation_files` - paths to `train.txt`, `val.txt`
*  `Config.tacatron_checkpoint`, `Config.waveglow_checkpoint` - paths to pretrained Tacotron2 and Waveglow,  
* `Config.output_directory` - path for writing checkpoints  


**Tacatron2**:

Set `Config.model_name` to `"Tacatron2"`  
* one gpu: 
`python train.py --exp default` 
* multigpu training:  
`python -m multiproc train.py --exp default`  
  

**WaveGlow**:

Set `Config.model_name` to `"WaveGlow"` 
* multigpu training:  
`python -m multiproc train.py --exp default`  
* one gpu:  
`python train.py --exp default`  

For restoring from checkpoint set path to checkpoints in `Config.restore_from`.  
In order to use emotions, set `use_emotions` as **True**.  
For balancing loss, set `use_loss_coefficients` as **True** and paths to dicts with coefficients in `emotion_coefficients` and `speaker_coefficients`.  


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

Models write loss

![Tensorboard Scalars](/img/tacotron-scalars.png)

and generate sample audio each epoch. Transcripts for audio are listed in `default.py/Config.phrases`

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

select adress with 127.0.01 and put it in the browser.  
In this case:  
`http://127.0.0.1:6006/?token=bbd413aef225c1394be3b9de144242075e651bea937eecce`  
Notebook opens in reed-only mode.  

This script takes
text as input and runs Tacotron 2 and then WaveGlow inference to produce an
audio file. It requires  pre-trained checkpoints from Tacotron 2 and WaveGlow
models and input text.  

Change paths to checkpoints of pretrained WaveGlow and Tacatron2  in the cell [2] in the `inference.ipynb`.  
Write a text to be displayed in the cell [7] in the `inference.ipynb`.  


## Parameters

In this section, we list the most important hyperparameters,
together with their default values that are used to train Tacotron 2 and
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

These scripts work at FP16 with n=1 frame per decoder step.  
* Make Decoder work with n > 1 frames per step
* Make training work at FP16. 
















