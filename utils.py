import numpy as np
import os
import vgg16
from keras.models import Model
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from pickle import dump, load
import keras.backend as K
import sys
import tensorflow as tf
import string
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, GRU, Embedding, Dropout,LSTM
from keras.optimizers import RMSprop, adam
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers.merge import add
from keras.utils import to_categorical

def load_file(path):
    with open(path, 'r') as f:
        return f.read()


'''load catpion files, build a dictionary'''


def load_captions(file):
    mapping = dict()
    for line in file.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue

        image_id, image_token = tokens[0], tokens[1:]
        image_id = image_id.split('#')[0]
        image_token = ' '.join(image_token)

        if image_id not in mapping:
            mapping[image_id] = list()

        mapping[image_id].append(image_token)
    return mapping


def clean_captions(caption_dic):
    punctions = list(string.punctuation)
    for keys, caption_list in caption_dic.items():
        for i in range(len(caption_list)):
            caption = caption_list[i]
            caption_token = nltk.word_tokenize(caption)
            caption_token = [cap.lower() for cap in caption_token if cap not in punctions]
            caption_list[i] = ' '.join(caption_token)


'''read a single image and convert it to array'''


def load_image(path, size=None):
    img = Image.open(path)
    if not size is None:
        img = img.resize(size=size)

    img = np.array(img)
    img = img / 255.0

    # Convert 2-dim gray-scale array to 3-dim RGB array.
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


'''extract image features'''


def extract_features(path):
    model = vgg16.VGG_16(weights_path='vgg16_weights.h5')
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    # print(model.summary())

    features = dict()

    for name in os.listdir(path):
        file_path = path + name
        img = image.load_img(file_path, target_size=(224,224))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # print(image)
        img = preprocess_input(img)


        feature = model.predict(img, verbose=0)

        # img_id = name.split('.')[0]
        features[name] = feature

        # print(">%s" %name)

    return features


def extract_feature(filename):
    model = vgg16.VGG_16(weights_path='vgg16_weights.h5')
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    # print(model.summary())

    img = image.load_img(filename, target_size=(224,224))
    img = image.img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = img / 255.0

    img = preprocess_input(img)

    feature = model.predict(img, verbose=0)

    # img_id = name.split('.')[0]

    return feature
#
# def extract_feature(feature_file, image_name):
#     for name, feature in feature_file.items():
#         if image_name == name:
#             return feature_file[name]




'''handling captions set of training/test/validation'''

def divide_train_test_captions(image_set, captions_set):
    mapping = dict()
    for img in image_set.splitlines():
        for caption in captions_set[img]:
            if img not in mapping:
                mapping[img] = list()

            mapping[img].append(caption)

    return mapping


def mark_dict(caption_dict):
    marked_dict = dict()
    for key, cap in caption_dict.items():
        caption_mark = ['ssss ' + caption + ' eeee' for caption in cap]

        if key not in marked_dict:
            marked_dict[key] = list()
        marked_dict[key].append(caption_mark[0])

    return marked_dict


def mark_captions(caption_dict):
    marked_set = []
    caption_set = []
    for key, cap in caption_dict.items():
        caption_mark = ['ssss ' + caption + ' eeee' for caption in cap]
        marked_set.append(caption_mark)
        caption_set.append(cap)

    return marked_set, caption_set


def flatten_dict(caption_lists):
    caption_list = [caption for captions in caption_lists for caption in captions]
    return caption_list


def print_process(count, max_count):
    pct_compelte = count / max_count

    msg = '\r- Progress: {0:.1%}'.format(pct_compelte)

    sys.stdout.write(msg)
    sys.stdout.flush()


'''get image features by batch size 32 :
   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/22_Image_Captioning.ipynb'''


def process_images(filenames, batch_size=32):
    vgg_model = vgg16.VGG_16(weights_path='vgg16_weights.h5')
    # print(vgg_model.summary())

    last_fc_layer = vgg_model.layers[-3]

    vgg_model_new = Model(inputs=vgg_model.input, outputs=last_fc_layer.output)

    input_img_size = K.int_shape(vgg_model.input)[1:3]
    transfer_img_size = K.int_shape(last_fc_layer.output)[1]

    num_images = len(filenames.splitlines())
    shape = (batch_size,) + input_img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float32)

    shape = (num_images, transfer_img_size)
    output_value = np.zeros(shape=shape, dtype=np.float32)

    start_index = 0

    while start_index < num_images:
        print_process(count=start_index, max_count=num_images)

        end_index = start_index + batch_size

        if end_index > num_images:
            end_index = num_images

        current_batch_size = end_index - start_index

        for i, filename in enumerate(filenames.splitlines()[start_index:end_index]):
            # path = os.path.join(data_dir, filename)

            img = load_image('Flickr8k/Flicker8k_Dataset/' + filename, size=input_img_size)
            image_batch[i] = img

        output_value_batch = vgg_model_new.predict(image_batch[0:current_batch_size])

        output_value[start_index:end_index] = output_value_batch[0:current_batch_size]

        start_index = end_index

    return output_value


def get_dataset_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset.splitlines()}
    return features


def generate_tokenizer(caption_list):
    # captions = []
    # for key, cap in caption_list.items():
    #     captions.append(cap)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(caption_list)
    return tokenizer


class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""

    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text

    def captions_to_tokens(self, captions_listlist):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """

        # Note that text_to_sequences() takes a list of texts.
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]

        return tokens


'''generate captions by choosing random tokens'''


def get_random_caption_tokens(idx, tokens_train):
    result = []

    for i in idx:
        # here select one of those token-sequences at random

        j = np.random.choice(len(tokens_train[i]))
        tokens = tokens_train[i][j]

        result.append(tokens)

    return result


# for creating ramdom batches of training-data
# quote: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/22_Image_Captioning.ipynb

def batch_generator(batch_size, train_features, train_num,tokens_train):
    while True:
        idx = np.random.randint(train_num, size=batch_size)

        transfer_values = train_features[idx]

        tokens = get_random_caption_tokens(idx,tokens_train)

        num_tokens = [len(t) for t in tokens]
        max_tokens = np.max(num_tokens)

        tokens_padded = pad_sequences(tokens, maxlen=max_tokens, padding='post', truncating='post')

        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]
        x_data = {'transfer_values_input': transfer_values,
                  'decoder_input': decoder_input_data
                  }

        y_data = {

            'decoder_output': decoder_output_data
        }

        yield (x_data, y_data)


'''for data generator'''
# def create_sequences (tok, maxlength, caption_list, photo, vocab_size):
#     X1, X2, y = [],[],[]
#     for caption in caption_list:
#         seq = (tok.texts_to_sequences([caption]))[0]
#         for i in range(1, len(seq)):
#             in_seq, out_seq = seq[:i], seq[i]
#             in_seq = pad_sequences([in_seq], maxlen=maxlength)[0]
#             out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
#
#             X1.append(photo)
#             X2.append(in_seq)
#             y.append(out_seq)
#
#     return np.array(X1), np.array(X2), np.array(y)
#
# def data_generator(marked_dict, photos, tok, maxlength, vocab_size):
#     while 1:
#         for key, caption_list in marked_dict.items():
#             photo = photos[key][0]
#             input_image, input_seq, out_text = create_sequences(tok, maxlength, caption_list, photo, vocab_size)
#             yield[[input_image, input_seq], out_text]


'''not using for generator'''

# def create_sequences(tok, maxlength, caption_dict, features, vocab_size):
#     X1, X2, y = [],[], []
#     for key, caption_list in caption_dict.items():
#         for caption in caption_list:
#             seq = (tok.texts_to_sequences([caption]))[0]
#             for i in range(1, len(seq)):
#                 in_seq, out_seq = seq[:i], seq[i]
#                 in_seq = pad_sequences([in_seq], maxlen=maxlength)[0]
#                 out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
#
#                 X1.append(features[key][0])
#                 X2.append(in_seq)
#                 y.append(out_seq)
#
#     return np.array(X1), np.array(X2), np.array(y)
#
#

def build_model(vocab_size, maxlength):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(maxlength,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation=(tf.nn.softmax))(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-4))
    # summarize model
    # model.summary()
    return model



def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.

    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# def generate_captions(img_name, maxlength):
#
#     image = load_image('Flickr8k/Flicker8k_Dataset/'+img_name, size =input_img_size)
#     expand_image = np.expand_dims(image, axis =0)
#     transfer_values = vgg_model_new.predict(expand_image)
#
#     shape = (1, maxlength)
#     decoder_input_data =np.zeros(shape= shape, dtype= np.int)
#
#     token_int = token_start
#
#     output_text = ' '
#
#     count_tokens =0
#
#
#     while token_int != token_end and count_tokens < maxlength:
#         decoder_input_data[0, count_tokens] = token_int
#
#         x_data ={
#             'transfer_values_input':transfer_values,
#             'decoder_input' : decoder_input_data
#         }
#
#         decoder_output = decoder_model.predict(x_data)
#
#         token_onehot = decoder_output[0, count_tokens, :]
#
#         token_int = np.argmax(token_onehot)
#         sample_word = tokenizer.token_to_word(token_int)
#
#         output_text += ' ' +sample_word
#
#         count_tokens += 1
#
#     output_tokens = decoder_input_data[0]
#
#     true_caption = img_tokens_dict[img_name]
#     return output_text, true_caption

# def word_for_id(integer, tok):
#     for word, index in tok.word_index.items():
#         if index == integer:
#             return word
#     return None
#
#
# def generate_captions(img_name, max_tokens):
#
#     image = load_image('Flickr8k/Flicker8k_Dataset/'+img_name, size =input_img_size)
#     expand_image = np.expand_dims(image, axis =0)
#     transfer_values = vgg_model_new.predict(expand_image)
#
#     text = 'ssss'
#
#     for i in range(max_tokens):
#         seq = tokenizer.texts_to_sequences([text])[0]
#         seq = pad_sequences([seq], maxlen=max_tokens)
#
#         y_pred = decoder_model.predict([transfer_values, seq], verbose=0)
#
#         y_pred = y_pred[0, i, :]
#
#         y_pred = np.argmax(y_pred)+1
#
#         word = word_for_id(y_pred, tokenizer)
#
#
#         if word is None:
#             break
#
#         text += ' '+ word
#
#         if word =='eeee':
#             break
#
#     true_caption = img_tokens_dict[img_name]
#     return text, true_caption
