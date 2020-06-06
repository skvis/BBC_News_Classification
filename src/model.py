import config
import utils
import tensorflow as tf
import joblib


def rnn():

    model = tf.keras.Sequential([
                tf.keras.layers.Embedding(config.VOCAB_SIZE,
                                          config.EMBEDDING_DIM,
                                          input_length=config.MAX_LENGTH),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(24, activation='relu'),
                tf.keras.layers.Dense(6, activation='softmax')])

    return model
