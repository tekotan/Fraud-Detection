# Code repository for parking garage loss prevention 

## ds/
All code related to model

## de/
### All code related to data engineering (mining)
- Use training_data_prep.py go generate fraud data
```bash
usage: training_data_prep.py [-h] -i REDSHIFT_TRANS_CSV_FNAME \
       -ml_f ML_FEATURES_COLUMNS_FNAME \
       -da_f DATA_AUG_FEATURES_COLUMNS_FNAME \
       -o OUTPUT_DIR
```
- Example
```bash
python3 training_data_prep.py -i /home/ajay/data/closedticket_04_01_2018_04_07_2018.csv -ml_f fheaders/ml_features_headers.txt -da_f fheaders/training_data_aug_headers.txt -o ./trn_data_out
```

## data/
Some sample data
