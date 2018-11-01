import pandas as pd
from datetime import datetime
import itertools

import tensorflow as tf

from utils import standard_scaler, maxmin_scaler
from data_prep import FdDataPrep
from fd_model import FdModel

PREDICT_DATA_FILE = 'data/data-01.csv'
DATA_SIZE = 2000

# Create data prep and model objects
fd_data_prep = FdDataPrep()
fd_model = FdModel()

input_fn = lambda: fd_data_prep.csv_input_fn(
  PREDICT_DATA_FILE,
  mode=tf.contrib.learn.ModeKeys.INFER,
  num_epochs=1,
  batch_size=500
)

estimator = fd_model.create_estimator(run_config, hparams)

predictions = estimator.predict(input_fn=input_fn)
predictions = itertools.islice(predictions, DATA_SIZE)
predictions = list(map(lambda item: list(item["encoding"]), predictions))

print(predictions[:5])

y = pd.read_csv(PREDICT_DATA_FILE, header=None, index_col=0)[65]

data_reduced = pd.DataFrame(predictions, columns=['c1','c2','c3'])
data_reduced['class'] = y
data_reduced.head()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=data_reduced.c2/1000000, ys=data_reduced.c3/1000000, zs=data_reduced.c1/1000000, c=data_reduced['class'], marker='o')
plt.show()


