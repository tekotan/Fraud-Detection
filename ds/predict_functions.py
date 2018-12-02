import pandas as pd
from datetime import datetime
import itertools
import numpy as np
import tensorflow as tf

from utils import standard_scaler, maxmin_scaler
from data_prep import FdDataPrep
from fd_model import FdModel
import config


def predict_classifier(predict_file):
    MODEL_NAME = config.CLASSIFIER_MODEL_NAME
    # MODEL_NAME = "dnn_classifier"
    model_dir = 'trained_models/{}'.format(MODEL_NAME)
    PREDICT_FILE = predict_file

    # hyper-params
    hparams = config.classifier_hparams

    # Create data prep and model objects
    fd_data_prep = FdDataPrep(PREDICT_FILE)
    fd_model = FdModel(fd_data_prep)

    predict_x = fd_data_prep.read_predict_data(PREDICT_FILE)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=fd_model.get_encodings(predict_x), num_epochs=1,
        batch_size=hparams.batch_size, shuffle=False
    )

    # Run config for training
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir
    )

    estimator = fd_model.create_linear_encoding_classifier(run_config, hparams)

    predictions = estimator.predict(input_fn=input_fn)
    formatted_pred = list(map(lambda x: list(x['classes']), predictions))
    print(formatted_pred)
    return formatted_pred


def test_auto_encoder(predict_file):
    fd_data_prep = FdDataPrep(predict_file)
    fd_model = FdModel(fd_data_prep)

    predict_x = fd_data_prep.read_predict_data(predict_file)
    print(fd_model.get_encodings(predict_x))
    return fd_model.get_encodings(predict_x)
