# Code repository for parking garage loss prevention

## ds/
All code related to model
### Config file
Contains all hyper parameter for training
### To train model
```bash
python full_train.py
```
### To predict and test model
```bash
python full_predict.py -m MODEL_TYPE -f PREDICT_FILE
```
## de/
### All code related to data engineering (mining)
- Use training_data_prep.py go generate fraud data
```bash
usage: training_data_prep.py [-h] -i REDSHIFT_TRANS_CSV_DIRNAME \
       -ml_f ML_FEATURES_COLUMNS_FNAME \
       -da_f DATA_AUG_FEATURES_COLUMNS_FNAME \
       -o OUTPUT_DIR
```
- Example
```bash
python3 training_data_prep.py -i /home/ajay/data/2018/closed_apr -ml_f fheaders/ml_features_headers.txt -da_f fheaders/training_data_aug_headers.txt -o ./trn_data_out
```
