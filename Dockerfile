FROM nvcr.io/nvidia/pytorch:19.06-py3

RUN apt-get update && \
    apt-get install -y \
        libasound-dev \
        portaudio19-dev \
        libportaudio2 \
        libportaudiocpp0 \
        ffmpeg \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

