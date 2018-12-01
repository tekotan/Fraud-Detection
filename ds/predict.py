import pandas as pd
from datetime import datetime
import itertools

import tensorflow as tf

from utils import standard_scaler, maxmin_scaler
from data_prep import FdDataPrep
from fd_model import FdModel

#MODEL_NAME = 'auto-encoder-02'
MODEL_NAME = "dnn_classifier"
model_dir = 'trained_models/{}'.format(MODEL_NAME)
PREDICT_FILE = '../data/test_prediction.csv'

# hyper-params
hparams  = tf.contrib.training.HParams(
  batch_size = 128,
  hidden_units=[3],
  l2_reg = 0.0000,
  noise_level = 0.0,
  dropout_rate = 0.00
)

# Create data prep and model objects
fd_data_prep = FdDataPrep(PREDICT_FILE)
fd_model = FdModel(fd_data_prep)

predict_x = fd_data_prep.read_predict_data(PREDICT_FILE)

input_fn=tf.estimator.inputs.pandas_input_fn(
  x=predict_x, num_epochs=1, \
  batch_size=hparams.batch_size, shuffle=False
)

# Run config for training
run_config = tf.estimator.RunConfig(
  tf_random_seed=19830610,
  model_dir=model_dir
)

estimator = fd_model.create_linear_classifier(run_config, hparams)

predictions = estimator.predict(input_fn=input_fn)
print('**** {}'.format(next(predictions)))

"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=data_reduced.c2/1000000, ys=data_reduced.c3/1000000, zs=data_reduced.c1/1000000, c=data_reduced['class'], marker='o')
plt.show()
"""
