import config
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import sys


def read_csv():
    df = pd.read_csv(config.DATA_PATH)
    df = df.sample(frac=1).reset_index(drop=True)
    labels = list(df['category'])
    sentences = list(df['text'])
    return sentences, labels


def split_dataset():
    sentences, labels = read_csv()

    train_size = int(len(sentences) * config.TRAIN_PORTION)
    train_sentences = sentences[0:train_size]
    train_labels = labels[0:train_size]
    valid_sentences = sentences[train_size:]
    valid_labels = labels[train_size:]

    return train_sentences, train_labels, valid_sentences, valid_labels, labels


def tokenizer_sequences():
    train_sentences, train_labels, valid_sentences, valid_labels, labels = split_dataset()
    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE,
                          oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences,
                                 maxlen=config.MAX_LENGTH,
                                 padding=config.PAD_TYPE,
                                 truncating=config.TRUNC_TYPE)

    valid_sequences = tokenizer.texts_to_sequences(valid_sentences)
    valid_padded = pad_sequences(valid_sequences,
                                 maxlen=config.MAX_LENGTH,
                                 padding=config.PAD_TYPE,
                                 truncating=config.TRUNC_TYPE)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    train_label_seq = label_tokenizer.texts_to_sequences(train_labels)
    valid_label_seq = label_tokenizer.texts_to_sequences(valid_labels)
    train_label_seq = np.array(train_label_seq)
    valid_label_seq = np.array(valid_label_seq)

    joblib.dump(tokenizer, f'{config.MODEL_PATH}tokenizer.pkl')
    joblib.dump(word_index, f"{config.MODEL_PATH}word_ind.pkl")
    joblib.dump(valid_padded, f"{config.MODEL_PATH}valid_padded.pkl")
    joblib.dump(valid_label_seq, f"{config.MODEL_PATH}valid_label_seq.pkl")

    return train_padded, train_label_seq, valid_padded, valid_label_seq


if __name__ == '__main__':
    tokenizer_sequences()
