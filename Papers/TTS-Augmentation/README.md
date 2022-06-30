# Checkpoints
  
| Experiment       |Checkpoints|
| ------------- |:------:|
| 1. Common Voice + TTS dataset |[Portuguese](https://huggingface.co/Edresson/wav2vec2-large-100k-voxpopuli-ft-Common-Voice_plus_TTS-Dataset-portuguese), [Russian](https://huggingface.co/Edresson/wav2vec2-large-100k-voxpopuli-ft-Common-Voice_plus_TTS-Dataset-russian)|
| 2. TTS dataset + TTS/VC Data Augmentation|[Portuguese](https://huggingface.co/Edresson/wav2vec2-large-100k-voxpopuli-ft-TTS-Dataset-plus-data-augmentation-portuguese), [Russian](https://huggingface.co/Edresson/wav2vec2-large-100k-voxpopuli-ft-TTS-Dataset-plus-data-augmentation-russian)|
| 3. Common Voice + TTS dataset + TTS/VC Data Augmentation|[Portuguese](https://huggingface.co/Edresson/wav2vec2-large-100k-voxpopuli-ft-Common_Voice_plus_TTS-Dataset_plus_Data_Augmentation-portuguese), [Russian](https://huggingface.co/Edresson/wav2vec2-large-100k-voxpopuli-ft-Common_Voice_plus_TTS-Dataset_plus_Data_Augmentation-russian)|


# Datasets
All dataset including generated is available at [Google drive](https://drive.google.com/drive/folders/1jRQI0fKuLDGMa5zfl_4VuPCmW3037y-Y?usp=sharing).

# Run example
```
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config_path Papers/TTS-Augmentation/configs/one-speaker/FINAL/GT/config_train_CV_PT.json --checkpoint_path "facebook/wav2vec2-large-100k-voxpopuli" > nogup-final-one-speaker-GT-PT.out 2>&1 &
