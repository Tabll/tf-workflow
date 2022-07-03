"""
Tensorflow 工作流测试代码

@docs https://www.tensorflow.org/text/guide/word_embeddings
"""
import os
import re
import shutil
import string

import tensorflow as tf
from keras.layers.preprocessing.text_vectorization import TextVectorization
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D

if __name__ == '__main__':
    url = "https://temp-1252262977.cos.ap-shanghai.myqcloud.com/data/test/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file(
        "aclImdb_v1.tar.gz", url, untar=True, cache_dir='.', cache_subdir=''
    )

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    batch_size = 1024
    seed = 123
    train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', batch_size=batch_size, validation_split=0.2,
        subset='training', seed=seed)
    val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', batch_size=batch_size, validation_split=0.2,
        subset='validation', seed=seed)

    for text_batch, label_batch in train_ds.take(1):
        for i in range(5):
            print(label_batch[i].numpy(), text_batch.numpy()[i])

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Embed a 1,000 word vocabulary into 5 dimensions.
    embedding_layer = tf.keras.layers.Embedding(1000, 5)

    # Create a custom standardization function to strip HTML break tags '<br />'.
    def custom_standardization(input_data):
        """
        :param input_data:
        :return:
        """
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation), '')

    # Vocabulary size and number of words in a sequence.
    vocab_size = 10000
    sequence_length = 100

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=vocab_size,
                                        output_sequence_length=sequence_length)

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    embedding_dim = 16

    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, name="embedding"),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[tensorboard_callback])

    model.save("./outputs/")
