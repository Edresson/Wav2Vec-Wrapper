FROM flml/flashlight:cuda-latest 

RUN apt-get update

# Install dependency
RUN apt install -y ffmpeg sox libsox-fmt-mp3 psmisc



RUN python3 -m pip install packaging 
RUN python3 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install gdown

WORKDIR /root/flashlight/bindings/python
RUN python3 setup.py install

# hugging face
RUN python3 -m pip install transformers datasets jiwer PyYAML librosa

WORKDIR /mnt/