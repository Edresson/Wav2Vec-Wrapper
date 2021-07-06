# Wav2Vec-Wrapper
An easy way to fine-tune Wav2Vec 2.0 for low-resource languages.

# Installation
Clone the repository.

```bash
git clone https://github.com/Edresson/Wav2Vec-Wrapper
pip3 install -r requeriments.txt
```

## Install Flashlight dependencies to use KenLM
### Use Docker: 
In the Wav2Vec-Wrapper repository execute:

```
nvidia-docker build ./ -t huggingface_flashlight
```

Now see the id of the docker image you just created: 
    
```
docker images
```
    
Using the IMAGE_ID run the command:
    
```
nvidia-docker run  --runtime=nvidia -v ~/:/mnt/ --rm  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --name wav2vec-wrapper -it IMAGE_ID bash
```

### Manually Instalation:
Please see the Flashlight documentation [here](https://github.com/flashlight/flashlight/tree/master/bindings/python#installation)

# Inference

You can easily run inference on a folder of wav files by runing:

```
python3 test.py --config_path ./example/config_eval.json --checkpoint_path_or_name facebook/wav2vec2-large-xlsr-53-french --audio_path ../wavs/ --no_kenlm
```

To run inference with a KenLM language model, you need to specify the apppropriate paths in the config file and remove the `--no_kenlm` flag.

To generate the `lexicon.lst` file, you can use the `./utils/generate-vocab.ipynb` notebook on your corpus.