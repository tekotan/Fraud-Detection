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

print(tf.__version__)

MODEL_NAME = 'auto-encoder-02'

TRAIN_DATA_FILE = '/home/ajay/code/external/tf-estimator-tutorials/05_Autoencoding/data/data-01.csv'

RESUME_TRAINING = False

MULTI_THREADING = True

FEATURE_COUNT = 64

HEADER = ['key']
HEADER_DEFAULTS = [[0]]
UNUSED_FEATURE_NAMES = ['key']
CLASS_FEATURE_NAME = 'CLASS'
FEATURE_NAMES = []  

for i in range(FEATURE_COUNT):
  HEADER += ['x_{}'.format(str(i+1))]
  FEATURE_NAMES += ['x_{}'.format(str(i+1))]
  HEADER_DEFAULTS += [[0.0]]

HEADER += [CLASS_FEATURE_NAME]
HEADER_DEFAULTS += [['NA']]

print("Header: {}".format(HEADER))
print("Features: {}".format(FEATURE_NAMES))
print("Class Feature: {}".format(CLASS_FEATURE_NAME))
print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))

def parse_csv_row(csv_row):
  
  columns = tf.decode_csv(csv_row, record_defaults=HEADER_DEFAULTS)
  features = dict(zip(HEADER, columns))
  
  for column in UNUSED_FEATURE_NAMES:
    features.pop(column)

  target = features.pop(CLASS_FEATURE_NAME)

  return features, target

def csv_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL, 
         skip_header_lines=0, 
         num_epochs=None, 
         batch_size=200):
  
  shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
  
  print("")
  print("* data input_fn:")
  print("================")
  print("Input file(s): {}".format(files_name_pattern))
  print("Batch size: {}".format(batch_size))
  print("Epoch Count: {}".format(num_epochs))
  print("Mode: {}".format(mode))
  print("Shuffle: {}".format(shuffle))
  print("================")
  print("")
  
  file_names = tf.matching_files(files_name_pattern)

  dataset = data.TextLineDataset(filenames=file_names)
  dataset = dataset.skip(skip_header_lines)
  
  if shuffle:
    dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    
  num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1
  
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda csv_row: parse_csv_row(csv_row), num_parallel_calls=num_threads)
  
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()
  
  features, target = iterator.get_next()

  return features, target

features, target = csv_input_fn(files_name_pattern="")
print("Feature read from CSV: {}".format(list(features.keys())))
print("Target read from CSV: {}".format(target))

df_params = pd.read_csv("data/params.csv", header=0, index_col=0)
len(df_params)
df_params['feature_name'] = FEATURE_NAMES
df_params.head()


def standard_scaler(x, mean, stdv):
  return (x-mean)/stdv

def maxmin_scaler(x, max_value, min_value):
  return (x-min_value)/(max_value-min_value)

def get_feature_columns():
  
  feature_columns = {}
  

#   feature_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
#            for feature_name in FEATURE_NAMES}

  for feature_name in FEATURE_NAMES:

    feature_max = df_params[df_params.feature_name == feature_name]['max'].values[0]
    feature_min = df_params[df_params.feature_name == feature_name]['min'].values[0]
    normalizer_fn = lambda x: maxmin_scaler(x, feature_max, feature_min)
    
    feature_columns[feature_name] = tf.feature_column.numeric_column(feature_name, 
                                     normalizer_fn=normalizer_fn
                                    )
 

  return feature_columns

print(get_feature_columns())

def autoencoder_model_fn(features, labels, mode, params):
  
  feature_columns = list(get_feature_columns().values())
  
  input_layer_size = len(feature_columns)
  
  encoder_hidden_units = params.encoder_hidden_units
  
  # decoder units are the reverse of the encoder units, without the middle layer (redundant)
  decoder_hidden_units = encoder_hidden_units.copy()  
  decoder_hidden_units.reverse()
  decoder_hidden_units.pop(0)
  
  output_layer_size = len(FEATURE_NAMES)
  
  he_initialiser = tf.contrib.layers.variance_scaling_initializer()
  l2_regulariser = tf.contrib.layers.l2_regularizer(scale=params.l2_reg)
  
  
  print("[{}]->{}-{}->[{}]".format(len(feature_columns)
                   ,encoder_hidden_units
                   ,decoder_hidden_units,
                   output_layer_size))

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  
  # input layer
  input_layer = tf.feature_column.input_layer(features=features, 
                        feature_columns=feature_columns)
  
  # Adding Gaussian Noise to input layer
  noisy_input_layer = input_layer + (params.noise_level * tf.random_normal(tf.shape(input_layer)))
  
  # Dropout layer
  dropout_layer = tf.layers.dropout(inputs=noisy_input_layer, 
                   rate=params.dropout_rate, 
                   training=is_training)

#   # Dropout layer without Gaussian Nosing
#   dropout_layer = tf.layers.dropout(inputs=input_layer, 
#                     rate=params.dropout_rate, 
#                     training=is_training)

  # Encoder layers stack
  encoding_hidden_layers = tf.contrib.layers.stack(inputs= dropout_layer,
                           layer= tf.contrib.layers.fully_connected,
                           stack_args=encoder_hidden_units,
                           #weights_initializer = he_init,
                           weights_regularizer =l2_regulariser,
                           activation_fn = tf.nn.relu
                          )
  # Decoder layers stack
  decoding_hidden_layers = tf.contrib.layers.stack(inputs=encoding_hidden_layers,
                           layer=tf.contrib.layers.fully_connected,        
                           stack_args=decoder_hidden_units,
                           #weights_initializer = he_init,
                           weights_regularizer =l2_regulariser,
                           activation_fn = tf.nn.relu
                          )
  # Output (reconstructed) layer
  output_layer = tf.layers.dense(inputs=decoding_hidden_layers, 
               units=output_layer_size, activation=None)
  
  # Encoding output (i.e., extracted features) reshaped
  encoding_output = tf.squeeze(encoding_hidden_layers)
  
  # Reconstruction output reshaped (for serving function)
  reconstruction_output =  tf.squeeze(tf.nn.sigmoid(output_layer))
  
  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    
    # Convert predicted_indices back into strings
    predictions = {
      'encoding': encoding_output,
      'reconstruction': reconstruction_output
    }
    export_outputs = {
      'predict': tf.estimator.export.PredictOutput(predictions)
    }
    
    # Provide an estimator spec for `ModeKeys.PREDICT` modes.
    return tf.estimator.EstimatorSpec(mode,
                      predictions=predictions,
                      export_outputs=export_outputs)
  
  # Define loss based on reconstruction and regularization
  
#   reconstruction_loss = tf.losses.mean_squared_error(tf.squeeze(input_layer), reconstruction_output) 
#   loss = reconstruction_loss + tf.losses.get_regularization_loss()
  
  reconstruction_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.squeeze(input_layer), logits=tf.squeeze(output_layer))
  loss = reconstruction_loss + tf.losses.get_regularization_loss()
             
  # Create Optimiser
  optimizer = tf.train.AdamOptimizer(params.learning_rate)

  # Create training operation
  train_op = optimizer.minimize(
    loss=loss, global_step=tf.train.get_global_step())

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
    "rmse": tf.metrics.root_mean_squared_error(
      tf.squeeze(input_layer), reconstruction_output)
  }
                           
  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                        loss=loss,
                        train_op=train_op,
                        eval_metric_ops=eval_metric_ops)
  return estimator_spec


def create_estimator(run_config, hparams):
  estimator = tf.estimator.Estimator(model_fn=autoencoder_model_fn, 
                  params=hparams, 
                  config=run_config)
  
  print("")
  print("Estimator Type: {}".format(type(estimator)))
  print("")

  return estimator

TRAIN_SIZE = 2000
NUM_EPOCHS = 1000
BATCH_SIZE = 100
NUM_EVAL = 10

TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS
CHECKPOINT_STEPS = int((TRAIN_SIZE/BATCH_SIZE) * (NUM_EPOCHS/NUM_EVAL))

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

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.contrib.learn.RunConfig(
  save_checkpoints_steps=CHECKPOINT_STEPS,
  tf_random_seed=19830610,
  model_dir=model_dir
)

print(hparams)
print("Model Directory:", run_config.model_dir)
print("")
print("Dataset Size:", TRAIN_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Steps per Epoch:",TRAIN_SIZE/BATCH_SIZE)
print("Total Steps:", TOTAL_STEPS)
print("Required Evaluation Steps:", NUM_EVAL) 
print("That is 1 evaluation step after each",NUM_EPOCHS/NUM_EVAL," epochs")
print("Save Checkpoint After",CHECKPOINT_STEPS,"steps")


train_spec = tf.estimator.TrainSpec(
  input_fn = lambda: csv_input_fn(
    TRAIN_DATA_FILE,
    mode = tf.contrib.learn.ModeKeys.TRAIN,
    num_epochs=hparams.num_epochs,
    batch_size=hparams.batch_size
  ),
  max_steps=hparams.max_steps,
  hooks=None
)

eval_spec = tf.estimator.EvalSpec(
  input_fn = lambda: csv_input_fn(
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

if not RESUME_TRAINING:
  print("Removing previous artifacts...")
  shutil.rmtree(model_dir, ignore_errors=True)
else:
  print("Resuming training...") 

  
tf.logging.set_verbosity(tf.logging.INFO)

time_start = datetime.utcnow() 
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................") 

estimator = create_estimator(run_config, hparams)

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

import itertools

DATA_SIZE = 2000

input_fn = lambda: csv_input_fn(
  TRAIN_DATA_FILE,
  mode=tf.contrib.learn.ModeKeys.INFER,
  num_epochs=1,
  batch_size=500
)

estimator = create_estimator(run_config, hparams)

predictions = estimator.predict(input_fn=input_fn)
predictions = itertools.islice(predictions, DATA_SIZE)
predictions = list(map(lambda item: list(item["encoding"]), predictions))

print(predictions[:5])

y = pd.read_csv(TRAIN_DATA_FILE, header=None, index_col=0)[65]

data_reduced = pd.DataFrame(predictions, columns=['c1','c2','c3'])
data_reduced['class'] = y
data_reduced.head()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=data_reduced.c2/1000000, ys=data_reduced.c3/1000000, zs=data_reduced.c1/1000000, c=data_reduced['class'], marker='o')
plt.show()


