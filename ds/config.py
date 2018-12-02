import tensorflow as tf

auto_encoder_hparams = tf.contrib.training.HParams(
    num_epochs=10,
    batch_size=128,
    hidden_units=[30, 3],
    learning_rate=0.001,
    l2_reg=0.0000,
    noise_level=0.0,
    max_steps=10000,
    dropout_rate=0.00,
)
classifier_hparams = tf.contrib.training.HParams(
    num_epochs=10,
    batch_size=128,
    hidden_units=[30, 3],
    learning_rate=0.001,
    l2_reg=0.0000,
    noise_level=0.0,
    max_steps=10000,
    dropout_rate=0.00,
)
model_features_list = [
    # 'locat',
    # 'ticketnum',
    # 'paycode',
    # 'make',
    # 'color',
    # 'plate',
    # 'ccdaccount',
    # 'ccdexpdate',
    # 'ratedescription',
    '_cardmultuses',
    'label',
]
TRAIN_DATA_FILE = "../de/trn_data_out/closed_apr/da_select_filtered_with_label_closed_apr.csv"
AUTO_ENCODER_MODEL_NAME = "auto-encoder-02"
AUTO_ENCODER_CHECKPOINT_STEPS = 200
RESUME_TRAINING = False

CLASSIFIER_MODEL_NAME = "complete_encoder_dnn_classifier"
CLASSIFIER_CHECKPOINT_STEPS = 200
