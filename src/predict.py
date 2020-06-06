import config
import data_preprocess
import joblib
import tensorflow as tf


def evaluate(load_model):
    valid_padded = joblib.load(f"{config.MODEL_PATH}valid_padded.pkl")
    valid_label_seq = joblib.load(f"{config.MODEL_PATH}valid_label_seq.pkl")
    score = load_model.evaluate(valid_padded, valid_label_seq)

    print('Accuracy:', score[1])
    print('Loss:', score[0])


if __name__ == '__main__':
    tokenizer = joblib.load(f'{config.MODEL_PATH}tokenizer.pkl')
    load_model = tf.keras.models.load_model(f"{config.MODEL_PATH}my_model.h5")
    evaluate(load_model)
