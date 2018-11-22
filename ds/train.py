from tensorflow.python import debug as tf_debug
import ipdb
from convert_fields_to_numericals import convert_select_columns_to_numericals
import sys
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
from tqdm import tqdm
from data_prep import FdDataPrep
from fd_model import FdModel

print(tf.__version__)

sys.path.append("../de")

#################################################
# Settings
#################################################
RESUME_TRAINING = False

TRAIN_DATA_FILE = "../de/trn_data_out/da_select_with_label_Orig_Data.csv"
TRAIN_SIZE = 100000
NUM_EPOCHS = 1000
BATCH_SIZE = 32
NUM_EVAL = 1
MODEL_NAME = "auto-encoder-02"
model_dir = "trained_models/{}".format(MODEL_NAME)
#################################################


# Create data prep and model objects
fd_data_prep = FdDataPrep()
fd_model = FdModel()

# features, target = fd_data_prep.csv_input_fn(files_name_pattern="data-*.csv")
# print("Feature read from CSV: {}".format(list(features.keys())))
# print("Target read from CSV: {}".format(target))

# print(fd_data_prep.get_feature_columns())

TOTAL_STEPS = (TRAIN_SIZE / BATCH_SIZE) * NUM_EPOCHS
CHECKPOINT_STEPS = int((TRAIN_SIZE / BATCH_SIZE) * (NUM_EPOCHS / NUM_EVAL))
print(TOTAL_STEPS)
# hyper-params
hparams = tf.contrib.training.HParams(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    encoder_hidden_units=[30, 3],
    learning_rate=0.01,
    l2_reg=0.0001,
    noise_level=0.0,
    max_steps=TOTAL_STEPS,
    dropout_rate=0.05,
)

# Run config for training
run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=32, tf_random_seed=19830610, model_dir=model_dir
)
# Print to verify
print(hparams)
print("Model Directory:", run_config.model_dir)
print("")
print("Dataset Size:", TRAIN_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Steps per Epoch:", TRAIN_SIZE / BATCH_SIZE)
print("Total Steps:", TOTAL_STEPS)
print("Required Evaluation Steps:", NUM_EVAL)
print("That is 1 evaluation step after each", NUM_EPOCHS / NUM_EVAL, " epochs")
print("Save Checkpoint After", CHECKPOINT_STEPS, "steps")

# Create train and eval specs
"""    input_fn=lambda: fd_data_prep.csv_input_fn(
        TRAIN_DATA_FILE,
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.batch_size,"""
# features, key = convert_select_columns_to_numericals(TRAIN_DATA_FILE)
features = np.array(pd.read_csv(
    TRAIN_DATA_FILE).dropna(how="any").get_values())
key = convert_select_columns_to_numericals(TRAIN_DATA_FILE)
head = [
    "s",
    "Unnamed: 0",
    "locat",
    "ticketnum",
    "dtout",
    "paycode",
    "make",
    "color",
    "plate",
    "ccdaccount",
    "ccdexpdate",
    "ratedescription",
    "label",
]
x = {}
for n, i in enumerate(head):
    if i == "s" or i == "dtout" or i == "Unnamed: 0":
        pass
    else:
        if i in key.keys():
            x[i] = features[:, n].astype(str)
        else:
            x[i] = features[:, n].astype(np.float32)
train_spec = tf.estimator.TrainSpec(
    input_fn=tf.estimator.inputs.numpy_input_fn(
        x, num_epochs=hparams.num_epochs, batch_size=hparams.batch_size, shuffle=True
    ),
    max_steps=hparams.max_steps,
    hooks=None,
)


eval_spec = tf.estimator.EvalSpec(
    input_fn=tf.estimator.inputs.numpy_input_fn(
        x, num_epochs=1, batch_size=hparams.batch_size, shuffle=False
    ),
    #   exporters=[tf.estimator.LatestExporter(
    #     name="encode",  # the name of the folder in which the model will be exported to under export
    #     serving_input_receiver_fn=csv_serving_input_fn,
    #     exports_to_keep=1,
    #     as_text=True)],
    #    steps=None,
    #    hooks=None,
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
tf.estimator.train_and_evaluate(estimator=estimator,
                                train_spec=train_spec, eval_spec=eval_spec)

time_end = datetime.utcnow()
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
