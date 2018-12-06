import shutil
from datetime import datetime

import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn import make_export_strategy
from tqdm import tqdm

from data_prep import FdDataPrep
from fd_model import FdModel
import config


def train_auto_encoder():
  #################################################
  # Settings
  #################################################
  RESUME_TRAINING = config.RESUME_TRAINING

  TRAIN_DATA_FILE = config.TRAIN_DATA_FILE
  # TRAIN_DATA_FILE = "./da_select_filtered_with_label_closed_apr.csv"
  MODEL_NAME = config.AUTO_ENCODER_MODEL_NAME
  # MODEL_NAME = "dnn_classifier"
  model_dir = "trained_models/{}".format(MODEL_NAME)
  #################################################

  # Create data prep and model objects
  fd_data_prep = FdDataPrep(TRAIN_DATA_FILE)
  fd_model = FdModel(fd_data_prep)
  # hyper-params
  hparams = config.auto_encoder_hparams

  # Run config for training
  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=config.AUTO_ENCODER_CHECKPOINT_STEPS, model_dir=model_dir)

  # Create train and eval specs
  train_x, train_y = fd_data_prep.read_training_data()

  train_spec = tf.estimator.TrainSpec(
      input_fn=tf.estimator.inputs.pandas_input_fn(
          x=train_x, num_epochs=hparams.num_epochs,
          batch_size=hparams.batch_size, shuffle=True
      ),
      max_steps=hparams.max_steps,
      hooks=None,
  )

  eval_spec = tf.estimator.EvalSpec(
      input_fn=tf.estimator.inputs.pandas_input_fn(
          train_x, num_epochs=1, batch_size=hparams.batch_size,
          shuffle=False
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
  print("Training started at {}".format(time_start.strftime("%H:%M:%S")))
  print(".......................................")

  # Create autoencoder estimator
  ae_estimator = fd_model.create_autoencoder_estimator(run_config, hparams)

  # Train model
  tf.estimator.train_and_evaluate(
      estimator=ae_estimator,
      train_spec=train_spec,
      eval_spec=eval_spec
  )

  time_end = datetime.utcnow()
  print(".......................................")
  print("Auto encoder training finished at {}".format(
      time_end.strftime("%H:%M:%S")))
  print("")
  time_elapsed = time_end - time_start
  print("Training elapsed time: {} seconds".format(
      time_elapsed.total_seconds()))


def train_classifier():
  #################################################
  # Settings
  #################################################
  RESUME_TRAINING = config.RESUME_TRAINING

  TRAIN_DATA_FILE = config.TRAIN_DATA_FILE
  MODEL_NAME = config.CLASSIFIER_MODEL_NAME
  model_dir = "trained_models/{}".format(MODEL_NAME)
  #################################################

  # Create data prep and model objects
  fd_data_prep = FdDataPrep(TRAIN_DATA_FILE)
  fd_model = FdModel(fd_data_prep)

  # Steps

  # hyper-params
  hparams = config.classifier_hparams

  # Run config for training
  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=config.CLASSIFIER_CHECKPOINT_STEPS, tf_random_seed=19830610, model_dir=model_dir
  )

  # Create train and eval specs
  train_x, train_y = fd_data_prep.read_training_data()

  train_spec = tf.estimator.TrainSpec(
      input_fn=tf.estimator.inputs.numpy_input_fn(
          x=fd_model.get_encodings(train_x), y=train_y, num_epochs=hparams.num_epochs,
          batch_size=hparams.batch_size, shuffle=True
      ),
      max_steps=hparams.max_steps,
      hooks=None,
  )

  eval_spec = tf.estimator.EvalSpec(
      input_fn=tf.estimator.inputs.numpy_input_fn(
          fd_model.get_encodings(train_x), train_y, num_epochs=1, batch_size=hparams.batch_size,
          shuffle=False
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
  print("Training started at {}".format(time_start.strftime("%H:%M:%S")))
  print(".......................................")

  # Create autoencoder estimator
  linear_classifier = fd_model.create_linear_encoding_classifier(
      run_config, hparams)

  # Train model
  tf.estimator.train_and_evaluate(
      # estimator=ae_estimator,
      estimator=linear_classifier,
      train_spec=train_spec,
      eval_spec=eval_spec
  )

  time_end = datetime.utcnow()
  print(".......................................")
  print("Classifier training finished at {}".format(
      time_end.strftime("%H:%M:%S")))
  print("")
  time_elapsed = time_end - time_start
  print("Training elapsed time: {} seconds".format(
      time_elapsed.total_seconds()))


def train_only_classifier():
  #################################################
  # Settings
  #################################################
  RESUME_TRAINING = config.RESUME_TRAINING

  TRAIN_DATA_FILE = config.TRAIN_DATA_FILE
  MODEL_NAME = config.ONLY_CLASSIFIER_MODEL_NAME
  model_dir = "trained_models/{}".format(MODEL_NAME)
  #################################################

  # Create data prep and model objects
  fd_data_prep = FdDataPrep(TRAIN_DATA_FILE)
  fd_model = FdModel(fd_data_prep)

  # Steps

  # hyper-params
  hparams = config.classifier_hparams

  # Run config for training
  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=config.CLASSIFIER_CHECKPOINT_STEPS, tf_random_seed=19830610, model_dir=model_dir
  )

  # Create train and eval specs
  train_x, train_y = fd_data_prep.read_training_data()

  train_spec = tf.estimator.TrainSpec(
      input_fn=tf.estimator.inputs.pandas_input_fn(
          x=train_x, y=train_y, num_epochs=hparams.num_epochs,
          batch_size=hparams.batch_size, shuffle=True
      ),
      max_steps=hparams.max_steps,
      hooks=None,
  )

  eval_spec = tf.estimator.EvalSpec(
      input_fn=tf.estimator.inputs.pandas_input_fn(
          train_x, train_y, num_epochs=1, batch_size=hparams.batch_size,
          shuffle=False
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
  print("Training started at {}".format(time_start.strftime("%H:%M:%S")))
  print(".......................................")

  # Create autoencoder estimator
  linear_classifier = fd_model.create_linear_classifier(
      run_config, hparams)

  # Train model
  tf.estimator.train_and_evaluate(
      # estimator=ae_estimator,
      estimator=linear_classifier,
      train_spec=train_spec,
      eval_spec=eval_spec
  )

  time_end = datetime.utcnow()
  print(".......................................")
  print("Classifier training finished at {}".format(
      time_end.strftime("%H:%M:%S")))
  print("")
  time_elapsed = time_end - time_start
  print("Training elapsed time: {} seconds".format(
      time_elapsed.total_seconds()))
