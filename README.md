# Tacotron 2 (multispeaker + gst) And WaveGlow

This repository provides a script and recipe to train Tacotron 2 and WaveGlow
v1.6 models to achieve state of the art accuracy, and is tested and maintained by NVIDIA.

## Table of Contents
* [Setup](##Setup)
   * [Requirements](###Requirements)
* Data Preprocessing (##Data preprocessing)
* [Training](##Training)
* [Model Description](#Model description)
   * [Scripts and sample code](#Scripts and sample code)
   * [Parameters](#Parameters)
      * [Shared parameters](#Shared parameters)
      * [Shared audio/STFT parameters](#Shared audio/STFT parameters)
      * [Tacotron 2 parameters](#Tacotron 2 parameters)
      * [WaveGlow parameters](#Waveglow parameters)
   

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

Run:

```bash
docker build --network=host --tag taco

sudo docker run --shm-size=8G --network=host --runtime=nvidia -v /absolute/path/to/your/code:/app -v 
/absolute/path/to/your/training_data:/data -v /absolute/path/to/your/logs:/logs -v /absolute/path/to/your/data:/raw-data -detach taco sleep inf
docker ps -a
```
Select Container ID of image `taco`. Run:

```
docker exec -it ID_of_taco_container bash
```

Clone the repository:
   ```bash
   git clone https://github.com/ide8/tacotron2_waveglow_multispeaker_gst
   cd tacotron2_waveglow_multispeaker_gst
   PROJDIR=$(pwd)
   export PYTHONPATH=$PROJDIR:$PYTHONPATH
   ```

## Data preprocessing


Run `preprocess.py`:   
`python preprocess.py --exp tacotron2`

 All necessary parameters for preprocessiong should be set in `./configs/__init__.py` in the class `PreprocessingConfig`.  
 If `start_from_preprocessed` flag is set to **False**, `preprocess.py` performs trimming of audio files up to `PreprocessingConfig.top_db` parameter, applies ffmpeg command (ffmpeg should be installed),
measures duration of audio files and lengths of correspondent text transcripts.  
 It saves a folder `wavs` with processed audio files and `data.csv` file in `PreprocessingConfig.output_directory` with the following format: `path_to_file, text, speaker_name, speaker_id, emotion, duration_of_audio, length_of_text`.  
Trimming and ffmpeg command are applied only to speakers, for which flag `process_audion` is **True**. Speakers with flag `emotion_present` is **False**, are treated as with emotion `neutral-normal`.  
If flag `start_from_preprocessed` is set to **True**, script loads file `data.csv` (it should be present in `PreprocessingConfig.output_directory`).  

Than script forms training and validation datasets in following way:
* selects rows with audio duration and text length less or equal those for speaker `PreprocessingConfig.limit_by`.  
If can't find, than it selects rows within `PreprocessingConfig.text_limit` and `PreprocessingConfig.dur_limit`. Lower limit for audio is defined by `PreprocessingConfig.minimum_viable_dur`. For being able to use the same batch size as NVIDIA guys, set `PreprocessingConfig.text_limit` to `linda_jonson`.
* splits dataset randomly by `ratio train : valid = 0.95 : 0.05`
* if train set is bigger than `PreprocessingConfig.n`, samples `n` rows
* saves `train.txt` and `val.txt` to `PreprocessingConfig.output_directory`
It saves `emotion_coefficients.json` and `speaker_coefficients.json` with coefficients for loss balancing (used by `train.py`).  

`PreprocessingConfig.cpus` defines number of cores to use when parallelize jobs.  
`PreprocessingConfig.sr` defines sample ratio for reading and writing audio.  
`PreprocessingConfig.emo_id_map` - dictionary for emotion mapping.  
`PreprocessingConfig.data['path']` is path to folder named with speaker name and containing `wavs` folder and `metadata.csv` with the following line format: `full_path_to_wav|text|emotion`.


## Training 
 In `./configs/__init__.py`, in the class `Config` set:
*  `training_files` and `validation_files` - paths to `train.txt`, `val.txt`
*  `Config.tacatron_checkpoint`, `Config.waveglow_checkpoint` - paths to pretrained Tacotron2 and Waveglow,  
* `Config.output_directory` - path for writing checkpoints  
and other training parameters such as learnig rate, batch size etc, that are listed below  

**Tacatron2**:

* multigpu training:  
`python -m multiproc train.py --exp tacotron2`  
* one gpu:  
`python train.py --exp tacotron2`  

**WaveGlow**:

* multigpu training:  
`python -m multiproc train.py --exp WaveGlow`  
* one gpu:  
`python train.py --exp WaveGlow`  

For restoring from checkpoint set path to checkpoints in `Config.restore_from`.  
In order to use emotions, set `use_emotions` as **True**.  
 For balancing loss, set `use_loss_coefficients` as **True** and paths to dicts with coefficients in `emotion_coefficients` and `speaker_coefficients`.  



## Repository description

### Scripts and sample code

The sample code for Tacotron 2 and WaveGlow has scripts specific to a
particular model, located in directories `./tacotron2` and `./waveglow`, as well as scripts common to both
models, located in the `./common` and `./router` directories. The model-specific scripts are as follows:

* `<model_name>/model.py` - the model architecture, definition of forward and
inference functions
* `<model_name>/data_function.py` - data loading functions
* `<model_name>/loss_function.py` - loss function for the model

The common scripts contain layer definitions common to both models
(`common/layers.py`), some utility scripts (`common/utils.py`) and scripts
for audio processing (`common/audio_processing.py` and `common/stft.py`). In
the root directory `./` of this repository, the `./train.py` script is used for
training . The
scripts `./models.py`, `./data_functions.py` and `./loss_functions.py` call
the respective scripts in the `<model_name>` directory, depending on what
model is trained using the `train.py` script. Scripts of the `./router` directory are used by `train.py` to select an appropriate model. 

### Parameters

In this section, we list the most important hyperparameters,
together with their default values that are used to train Tacotron 2 and
WaveGlow models.

#### Shared parameters

* `epochs` - number of epochs (Tacotron 2: 1701, WaveGlow: 1001)
* `learning-rate` - learning rate (Tacotron 2: 1e-3, WaveGlow: 1e-4)
* `batch-size` - batch size (Tacotron 2 64, WaveGlow: 10/4)
* `amp-run` - use mixed precision training

#### Shared audio/STFT parameters

* `sampling-rate` - sampling rate in Hz of input and output audio (22050)
* `filter-length` - (1024)
* `hop-length` - hop length for FFT, i.e., sample stride between consecutive FFTs (256)
* `win-length` - window size for FFT (1024)
* `mel-fmin` - lowest frequency in Hz (0.0)
* `mel-fmax` - highest frequency in Hz (8.000)

#### Tacotron 2 parameters

* `anneal-steps` - epochs at which to anneal the learning rate (700 1200 1700)
* `anneal-factor` - factor by which to anneal the learning rate (0.1)

#### WaveGlow parameters

* `segment-length` - segment length of input audio processed by the neural network (8000)



### Inference process

You can run inference using the `./inference.ipynb` script.  

Run Jupyter Notebook:  
`jupyter notebook --ip 0.0.0.0 --port 6006 --no-browser --allow-root`  

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

select adress with 127.0.01 and put it in the brouser.  
In this case:  
`http://127.0.0.1:6006/?token=bbd413aef225c1394be3b9de144242075e651bea937eecce`  
Notebook opens in reed-only mode.  

This script takes
text as input and runs Tacotron 2 and then WaveGlow inference to produce an
audio file. It requires  pre-trained checkpoints from Tacotron 2 and WaveGlow
models and input text as a text file, with one phrase per line.  

 In cell [2] change paths to checkpoints of pretrained WaveGlow and Tacatron2.  
In cell [7] write a text to be displayed.






