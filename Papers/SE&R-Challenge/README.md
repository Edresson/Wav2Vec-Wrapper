# SE&R Challenge

## Download CORAA dataset
Create the Directory for downloading the dataset:
```
    mkdir -p ../datasets/CORAA_Dataset/
    cd ../datasets/CORAA_Dataset/
```

### Download the Audios

#### Train 
    ```gdown --id 1deCciFD35EA_OEUl0MrEDa7u5O2KgVJM -O train.zip
    unzip train.zip
    ```
#### Development 
    ```
    gdown --id 1bIHctanQjW2ITOM5wNQSt_NjB45s0_Q_ -O dev.zip
    unzip dev.zip
    ```
### Transcriptions
#### Train 
    ```
    gdown --id 1HbwahfMWoArYj0z2PfI4dHiambWfaNWg -O metadata_train.csv
    ```
#### Development 
    ```
    gdown --id 185erjax7lS_YNuolZvcMt_EdprafyMU0 -O metadata_dev.csv
    ```

## Train the model using Transfer Learning from the baseline
 To perform a fine tuning using the baseline checkpoint, you must first adjust the configuration file paths. We provide an example configuration file at: "Papers/SE&R-Challenge/configs/config_coraa.json".
  Change the "output_path" to the path where you want the model to save the checkpoints during training and change the dataset paths. After just run the command:

``` 
CUDA_VISIBLE_DEVICES=0 python3 train.py --config_path Papers/SE&R-Challenge/configs/config_coraa.json --checkpoint_path Edresson/wav2vec2-large-xlsr-coraa-portuguese 

``` 


## Generate the transcriptions for the test set


To get the transcripts for the test set use the command below. Change "Papers/SE&R-Challenge/configs/config_coraa.json" to the config used in training your model, "../datasets/CORAA_Dataset/test/" to the path where the audio files for the test are, "Edresson/wav2vec2-large-xlsr-coraa-portuguese" to the path of your model. After executing the command the transcripts will be saved in the file " ../results/coraa_test_pred_transcriptions.csv" and you can submit them in the submission system.


``` 
CUDA_VISIBLE_DEVICES=0 python3 test.py --config_path Papers/SE&R-Challenge/configs/config_coraa.json --audio_path ../datasets/CORAA_Dataset/test/ --checkpoint_path_or_name Edresson/wav2vec2-large-xlsr-coraa-portuguese  --no_kenlm --output_csv  ../results/coraa_test_pred_transcriptions.csv

``` 

You can still evaluate your model in the development subset the same way it is done in the submission system using the command below where "../datasets/CORAA_Dataset/metadata_dev_final.csv" is the development CSV of the CORAA dataset and "../results/coraa_dev_pred_transcriptions.csv" is the CSV with the transcripts predicted by your model for the development subset. Enter in the directory "Papers/SE&R-Challenge/scripts/" and run:


``` 
python3 compute_metrics.py --dataset_csv ../datasets/CORAA_Dataset/metadata_dev_final.csv --transcription_csv ../results/coraa_dev_pred_transcriptions.csv

```