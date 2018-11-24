import shutil
from datetime import datetime

import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn import make_export_strategy
from tqdm import tqdm

from data_prep import FdDataPrep
from fd_model import FdModel

#################################################
# Settings
#################################################
RESUME_TRAINING = False

TRAIN_DATA_FILE = "../de/trn_data_out/closed_apr/da_select_filtered_with_label_closed_apr.csv"
#TRAIN_DATA_FILE = "./da_select_filtered_with_label_closed_apr.csv"
TRAIN_SIZE = 100000
NUM_EPOCHS = 1000
BATCH_SIZE = 32
NUM_EVAL = 1
MODEL_NAME = "auto-encoder-02"
model_dir = "trained_models/{}".format(MODEL_NAME)
#################################################


# Create data prep and model objects
fd_data_prep = FdDataPrep(TRAIN_DATA_FILE)
fd_model = FdModel(fd_data_prep)

# Steps
TOTAL_STEPS = (TRAIN_SIZE / BATCH_SIZE) * NUM_EPOCHS
CHECKPOINT_STEPS = int((TRAIN_SIZE / BATCH_SIZE) * (NUM_EPOCHS / NUM_EVAL))
print(TOTAL_STEPS)

# hyper-params
hparams = tf.contrib.training.HParams(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    encoder_hidden_units=[100, 30, 3],
    learning_rate=0.001,
    l2_reg=0.0000,
    noise_level=0.0,
    max_steps=TOTAL_STEPS,
    dropout_rate=0.00,
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
train_x = fd_data_prep.read_ae_training_data()

train_spec = tf.estimator.TrainSpec(
    input_fn=tf.estimator.inputs.pandas_input_fn(
        train_x, num_epochs=hparams.num_epochs, \
            batch_size=hparams.batch_size, shuffle=True
    ),
    max_steps=hparams.max_steps,
    hooks=None,
)

eval_spec = tf.estimator.EvalSpec(
    input_fn=tf.estimator.inputs.pandas_input_fn(
        train_x, num_epochs=1, batch_size=hparams.batch_size, shuffle=False
    ),
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
