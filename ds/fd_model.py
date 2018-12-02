import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn import make_export_strategy
from data_prep import FdDataPrep
import numpy as np


class FdModel(object):
    def __init__(self, data_prep):
        self.data_prep = data_prep

    def autoencoder_model_fn(self, features, labels, mode, params):
        feature_columns = list(self.data_prep.get_feature_columns().values())

        input_layer_size = len(feature_columns)

        hidden_units = params.hidden_units

        # decoder units are the reverse of the encoder units, without the middle layer (redundant)
        decoder_hidden_units = hidden_units.copy()
        decoder_hidden_units.reverse()
        decoder_hidden_units.pop(0)

        he_initialiser = tf.contrib.layers.variance_scaling_initializer()
        l2_regulariser = tf.contrib.layers.l2_regularizer(scale=params.l2_reg)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # input layer
        input_layer = tf.feature_column.input_layer(
            features=features, feature_columns=feature_columns
        )
        output_layer_size = int(input_layer.get_shape()[1])
        print(
            "[{}]->{}-{}->[{}]".format(
                len(feature_columns),
                hidden_units,
                decoder_hidden_units,
                output_layer_size,
            )
        )
        # Adding Gaussian Noise to input layer
        #    noisy_input_layer = input_layer + (params.noise_level * tf.random_normal(tf.shape(input_layer)))

        # Dropout layer
        #    dropout_layer = tf.layers.dropout(inputs=noisy_input_layer,
        #                     rate=params.dropout_rate,
        #                     training=is_training)

        # Dropout layer without Gaussian Nosing
        dropout_layer = tf.layers.dropout(
            inputs=input_layer, rate=params.dropout_rate, training=is_training
        )

        # Encoder layers stack
        encoding_hidden_layers = tf.contrib.layers.stack(
            inputs=dropout_layer,
            layer=tf.contrib.layers.fully_connected,
            stack_args=hidden_units,
            weights_initializer=he_initialiser,
            weights_regularizer=l2_regulariser,
            activation_fn=tf.nn.relu,
        )
        # Decoder layers stack
        decoding_hidden_layers = tf.contrib.layers.stack(
            inputs=encoding_hidden_layers,
            layer=tf.contrib.layers.fully_connected,
            stack_args=decoder_hidden_units,
            weights_initializer=he_initialiser,
            weights_regularizer=l2_regulariser,
            activation_fn=tf.nn.relu,
        )
        # Output (reconstructed) layer
        output_layer = tf.layers.dense(
            inputs=decoding_hidden_layers, units=output_layer_size, activation=None
        )

        # Encoding output (i.e., extracted features) reshaped
        encoding_output = tf.squeeze(encoding_hidden_layers)

        # Reconstruction output reshaped (for serving function)
        reconstruction_output = tf.squeeze(tf.nn.sigmoid(output_layer))

        # Provide an estimator spec for `ModeKeys.PREDICT`.
        if mode == tf.estimator.ModeKeys.PREDICT:

            # Convert predicted_indices back into strings
            predictions = {
                "encoding": encoding_output,
                "reconstruction": reconstruction_output,
            }
            export_outputs = {
                "predict": tf.estimator.export.PredictOutput(predictions)}

            # Provide an estimator spec for `ModeKeys.PREDICT` modes.
            return tf.estimator.EstimatorSpec(
                mode, predictions=predictions, export_outputs=export_outputs
            )

        # Define loss based on reconstruction and regularization

        reconstruction_loss = tf.losses.mean_squared_error(
            tf.squeeze(input_layer), reconstruction_output)
        loss = reconstruction_loss + tf.losses.get_regularization_loss()
        # reconstruction_loss = tf.losses.sigmoid_cross_entropy(
        # multi_class_labels=tf.squeeze(input_layer), logits=tf.squeeze(output_layer)
        # )
        #loss = reconstruction_loss + tf.losses.get_regularization_loss()

        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(params.learning_rate)

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Calculate root mean squared error as additional eval metric
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(
                tf.squeeze(input_layer), reconstruction_output
            )
        }
        if is_training:
            pass
        else:
            pass
        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops
        )
        return estimator_spec

    def create_autoencoder_estimator(self, run_config, hparams):
        estimator = tf.estimator.Estimator(
            model_fn=self.autoencoder_model_fn, params=hparams, config=run_config
        )

        print("")
        print("Estimator Type: {}".format(type(estimator)))
        print("")

        return estimator

    def create_linear_classifier(self, run_config, hparams):
        feature_columns = list(self.data_prep.get_feature_columns().values())
        params = hparams
        estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                               hidden_units=params.hidden_units,
                                               n_classes=2, config=run_config)

        print("")
        print("Estimator Type: {}".format(type(estimator)))
        print("")

        return estimator

    def create_linear_encoding_classifier(self, run_config, hparams):
        feature_columns = list(
            [tf.feature_column.numeric_column("x_" + str(i)) for i in range(3)])
        params = hparams
        estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                               hidden_units=params.hidden_units,
                                               n_classes=2, config=run_config)

        print("")
        print("Estimator Type: {}".format(type(estimator)))
        print("")

        return estimator

    def get_encodings(self, train_x):
        MODEL_NAME = 'auto-encoder-02'
        # MODEL_NAME = "dnn_classifier"
        model_dir = 'trained_models/{}'.format(MODEL_NAME)

        # hyper-params
        hparams = tf.contrib.training.HParams(
            batch_size=128,
            hidden_units=[30, 3],
            l2_reg=0.0000,
            noise_level=0.0,
            dropout_rate=0.00
        )

        # Create data prep and model objects
        input_fn = tf.estimator.inputs.pandas_input_fn(
            x=train_x, num_epochs=1,
            batch_size=hparams.batch_size, shuffle=False
        )

        # Run config for training
        run_config = tf.estimator.RunConfig(
            tf_random_seed=19830610,
            model_dir=model_dir
        )
        encodings = np.array(list(map(lambda x: x['encoding'], self.create_autoencoder_estimator(
            run_config, hparams).predict(input_fn=input_fn))))
        encoding_x = {}
        for i in range(hparams.hidden_units[-1]):
            encoding_x["x_" + str(i)] = encodings[:, i]
        return encoding_x
