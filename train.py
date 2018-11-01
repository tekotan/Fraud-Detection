import pandas as pd
import numpy as np
import shutil
import multiprocessing
from datetime import datetime

import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn import make_export_strategy
from tensorflow import data

from data_prep import FdDataPrep
from fd_model import FdModel

print(tf.__version__)

#################################################
# Settings
#################################################
RESUME_TRAINING = False

TRAIN_DATA_FILE = 'data/data-01.csv'
TRAIN_SIZE = 2000
NUM_EPOCHS = 10
BATCH_SIZE = 100
NUM_EVAL = 10
MODEL_NAME = 'auto-encoder-02'
model_dir = 'trained_models/{}'.format(MODEL_NAME)
#################################################


# Create data prep and model objects
fd_data_prep = FdDataPrep()
fd_model = FdModel()

features, target = fd_data_prep.csv_input_fn(files_name_pattern="")
print("Feature read from CSV: {}".format(list(features.keys())))
print("Target read from CSV: {}".format(target))

print(fd_data_prep.get_feature_columns())

TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS
CHECKPOINT_STEPS = int((TRAIN_SIZE/BATCH_SIZE) * (NUM_EPOCHS/NUM_EVAL))

# hyper-params
hparams  = tf.contrib.training.HParams(
  num_epochs = NUM_EPOCHS,
  batch_size = BATCH_SIZE,
  encoder_hidden_units=[30,3],
  learning_rate = 0.01,
  l2_reg = 0.0001,
  noise_level = 0.0,
  max_steps = TOTAL_STEPS,
  dropout_rate = 0.05
)

# Run config for training
run_config = tf.estimator.RunConfig(
  save_checkpoints_steps=CHECKPOINT_STEPS,
  tf_random_seed=19830610,
  model_dir=model_dir
)

# Print to verify 
print(hparams)
print("Model Directory:", run_config.model_dir)
print("")
print("Dataset Size:", TRAIN_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Steps per Epoch:", TRAIN_SIZE/BATCH_SIZE)
print("Total Steps:", TOTAL_STEPS)
print("Required Evaluation Steps:", NUM_EVAL) 
print("That is 1 evaluation step after each", NUM_EPOCHS/NUM_EVAL, " epochs")
print("Save Checkpoint After", CHECKPOINT_STEPS, "steps")

# Create train and eval specs 
train_spec = tf.estimator.TrainSpec(
  input_fn = lambda: fd_data_prep.csv_input_fn(
    TRAIN_DATA_FILE,
    mode = tf.contrib.learn.ModeKeys.TRAIN,
    num_epochs=hparams.num_epochs,
    batch_size=hparams.batch_size
  ),
  max_steps=hparams.max_steps,
  hooks=None
)

eval_spec = tf.estimator.EvalSpec(
  input_fn = lambda: fd_data_prep.csv_input_fn(
    TRAIN_DATA_FILE,
    mode=tf.contrib.learn.ModeKeys.EVAL,
    num_epochs=1,
    batch_size=hparams.batch_size
  ),
  #   exporters=[tf.estimator.LatestExporter(
  #     name="encode",  # the name of the folder in which the model will be exported to under export
  #     serving_input_receiver_fn=csv_serving_input_fn,
  #     exports_to_keep=1,
  #     as_text=True)],
  steps=None,
  hooks=None
)

# Remove previous artifacts if no resume_trainig
if not RESUME_TRAINING:
  print("Removing previous artifacts...")
  shutil.rmtree(model_dir, ignore_errors=True)
else:
  print("Resuming training...") 

# set logging
tf.logging.set_verbosity(tf.logging.INFO)

time_start = datetime.utcnow() 
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................") 

# Create estimator
estimator = fd_model.create_estimator(run_config, hparams)

# Train model
tf.estimator.train_and_evaluate(
  estimator=estimator,
  train_spec=train_spec, 
  eval_spec=eval_spec
)

time_end = datetime.utcnow() 
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
