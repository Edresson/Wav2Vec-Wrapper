# Wav2Vec-Wrapper
An easy way for fine-tuning the Wav2Vec2 for languages with low resources available.

# Installation
Clone the repository.

```bash
git clone https://github.com/Edresson/Wav2Vec-Wrapper
pip3 install -r requeriments.txt
```

## Install Flashlight dependencies for use the KenLM
### Use Docker: 
In the Wav2Vec-Wrapper repository execute:

    ```bash
    nvidia-docker build ./ -t huggingface_flashlight
    ```
Now see the id of the docker image you just created: 
    
    ```bash
    docker images
    ```
    
Using the IMAGE_ID run the command:
    
    ```bash
    nvidia-docker run  --runtime=nvidia -v ~/:/mnt/ --rm  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --name wav2vec-wrapper -it IMAGE_ID bash
    
    ```

### Manually Instalation:
Please see the Flashlight documentation [here](https://github.com/flashlight/flashlight/tree/master/bindings/python#installation)