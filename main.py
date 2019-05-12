import numpy as np
import glob
import os
from keras.preprocessing.text import Tokenizer
from keras.applications import inception_resnet_v2
import cv2
from keras.models import Model
from keras import backend as K
from pickle import load, dump
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from cache import cache
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, GRU, Embedding
from keras.optimizers import RMSprop, adam, SGD
from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import *
from tensorflow.python.ops import control_flow_ops
from keras.models import load_model


'''to solver the maximum_iterations error'''

orig_while_loop = control_flow_ops.while_loop


def patched_while_loop(*args, **kwargs):
    kwargs.pop("maximum_iterations", None)  # Ignore.
    return orig_while_loop(*args, **kwargs)


control_flow_ops.while_loop = patched_while_loop


'''load image files'''

train_img_read = load_file('Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt')
test_img_read = load_file('Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt')
val_img_read = load_file('Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt')

train_img_num = len(train_img_read.splitlines())
test_img_num = len(test_img_read.splitlines())
val_img_num = len(val_img_read.splitlines())


img_dataset = [os.path.basename(x) for x in glob.glob('Flickr8k/Flicker8k_Dataset/*.jpg')]

print('The number of training images: ', train_img_num)
print('The number of test images: ', test_img_num)
print('The number of validation images', val_img_num)


'''load caption file'''
img_captions = load_file('Flickr8k/Flickr8k_text/Flickr8k.token.txt')
img_tokens_dict = load_captions(img_captions)

# test = list(img_tokens.keys())[:1]
# cr = cv2.imread('Flickr8k/Flicker8k_Dataset/1000268201_693b08cb0e.jpg')
#
# plt.imshow(cr)
# plt.show()

clean_captions(img_tokens_dict)

word_list = set()
for key in img_tokens_dict.keys():
    [word_list.update(d.split()) for d in img_tokens_dict[key]]
    # [caption for caption in img_tokens_dict[key]]

print('The length of whole word list: ', len(word_list))
print()


'''mark the caption's sequence'''

start_marker = 'ssss '
end_marker = ' eeee'


train_token_dict = divide_train_test_captions(train_img_read, img_tokens_dict)

test_token_dict = divide_train_test_captions(test_img_read, img_tokens_dict)

val_token_dict = divide_train_test_captions(val_img_read, img_tokens_dict)

'''mark captions with sss and eee, return as list'''

train_captions_marked, train_captions = mark_captions(train_token_dict)
val_captions_marked, val_captions = mark_captions(val_token_dict)
test_captions_marked, test_captions = mark_captions(test_token_dict)
print(train_captions_marked[0])
print(train_captions[0])
print()

'''mark captions with sss and eee, return as dictionary'''

train_token_dict_marked = mark_dict(train_token_dict)
val_token_dict_marked = mark_dict(val_token_dict)
test_token_dict_marked = mark_dict(test_token_dict)


'''flatten the marked caption sets, return as list'''


def flatten_dict(caption_lists):
    caption_list = [caption for captions in caption_lists for caption in captions]
    return caption_list


train_captions_flat = flatten_dict(train_captions_marked)
test_captions_flat = flatten_dict(test_captions_marked)
val_captions_flat = flatten_dict(val_captions_marked)


'''embedding the caption set using keras.tokenizer'''

# here i selected 3000 most popular words in this dataset


num_words = 2000
# tokenizer = Tokenizer(num_words=num_words)
tokenizer = TokenizerWrap(texts=train_captions_flat, num_words=num_words)


token_start = tokenizer.word_index[start_marker.strip()]
print(token_start)
print()

token_end = tokenizer.word_index[end_marker.strip()]
print(token_end)
print()

train_token = tokenizer.captions_to_tokens(train_captions_marked)
test_token = tokenizer.captions_to_tokens(train_captions_marked)
val_token = tokenizer.captions_to_tokens(train_captions_marked)

print(train_token[0])
print()


max_length = max(len(d.split()) for d in train_captions_flat)
print('The max length of a sentence:', max_length)
print()


'''use a pre-trained GLOVE model to embed words'''










'''extract model from train/test/validation'''
#
# img_path = 'Flickr8k/Flicker8k_Dataset/'
# img_feature = extract_features(img_path)
# print('Extracted Features: %d' % len(img_feature))
# dump(img_feature,open('img_features.pkl', 'wb'))

# train_features = get_dataset_features('img_features.pkl',train_img_read)
# test_features = get_dataset_features('img_features.pkl', test_img_read)


def process_images_train():
    print("Processing {0} images in training-set ...".format(train_img_num))

    # Path for the cache-file.
    cache_path = os.path.join('./pickle/', "transfer_values_train.pkl")

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            filenames=train_img_read)

    return transfer_values


transfer_values_train = process_images_train()
print("dtype:", transfer_values_train.dtype)
print("shape:", transfer_values_train.shape)
print()


def process_images_val():
    print("Processing {0} images in validation-set ...".format(val_img_num))

    # Path for the cache-file.
    cache_path = os.path.join('./pickle/', "transfer_values_val.pkl")

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            filenames=val_img_read)

    return transfer_values


transfer_values_val = process_images_val()
print("dtype:", transfer_values_val.dtype)
print("shape:", transfer_values_val.shape)
print()


# def process_images_test():
#     print("Processing {0} images in test-set ...".format(test_img_num))
#
#     cache_path = os.path.join('./pickle/', 'transfer_values_test.pkl')
#
#     transfer_values = cache(cache_path=cache_path,
#                             fn=process_images,
#                             filenames=test_img_read)
#
#     return transfer_values
#
#
# transfer_values_test = process_images_test()
# print("dtype:", transfer_values_test.dtype)
# print("shape:", transfer_values_test.shape)
# print()


vgg_model = vgg16.VGG_16(weights_path='vgg16_weights.h5')
# print(vgg_model.summary())

last_fc_layer = vgg_model.layers[-3]

vgg_model_new = Model(inputs=vgg_model.input, outputs=last_fc_layer.output)

input_img_size = K.int_shape(vgg_model.input)[1:3]
transfer_img_size = K.int_shape(last_fc_layer.output)[1]

batch_size = 64


'''set the steps per epoch'''
num_captions_train = [len(cap) for cap in train_captions]
total_num_caption_train = np.sum(num_captions_train)
steps_per_epoch = int(total_num_caption_train / batch_size)
print('Steps per epoch: ', steps_per_epoch)


'''creat RNN'''
state_size = 512
embedding_size = 128
transfer_values_input = Input(shape=(transfer_img_size, ), name='transfer_values_input')

decoder_transfer_map = Dense(state_size, activation='tanh', name='decoder_transfer_map')

decoder_input = Input(shape=(None, ), name='decoder_input')

decoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_size, name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1', return_sequences=True)

fe1 = Dropout(0.5)
decoder_gru2 = GRU(state_size, name='decoder_gru2', return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3', return_sequences=True)
fe2 = Dropout(0.5)

decoder_dense = Dense(num_words, activation='linear', name='decoder_output')



def connect_decoder(transfer_values):
    initial_state = decoder_transfer_map(transfer_values)

    net = decoder_input

    # conect the embedding_layer
    net = decoder_embedding(net)

    # connect all the GRU layers
    net = decoder_gru1(net, initial_state=initial_state)
    net = fe1(net)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    net = fe2(net)
    decoder_output = decoder_dense(net)

    return decoder_output



decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input], outputs=[decoder_output])




''' Using Data generator'''

# generator = data_generator(train_token_dict_marked, train_features,
#                            tokenizer, max_length, len(word_list))
#
# inputs, outputs = next(generator)
#
# print(inputs[0])
# print(inputs[1])

# generator = batch_generator(512, transfer_values_train, train_img_num, train_token)
#
#
# batch = next(generator)
# batch_x = batch[0]
# batch_y = batch[1]
#
# print(batch_x['transfer_values_input'][0])
#
# print(batch_x['decoder_input'][0])
#
# print(batch_y['decoder_output'][0])
#
#


'''some test here'''

# decoder_model = build_model(len(word_list), max_length)

'''try different optimizer'''
# optimizer = RMSprop(lr=1e-5)
# optimizer = SGD(lr=1e-5)
optimizer = 'adam'
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))



decoder_model.compile(optimizer=optimizer, loss=sparse_cross_entropy, target_tensors=[decoder_target])
print(decoder_model.summary())

# path_checkpoint = 'checkpoint.keras'
# callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, verbose=1, save_weights_only=True)
# callback_tensorboard =TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=False)
# callback_earlystop = EarlyStopping()



'''load checkpoint'''
# try:
#     decoder_model.load_weights(path_checkpoint)
# except Exception as error:
#     print('Error trying to load checkpoint')
#     print(error)

'''for batch generator'''
# decoder_model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch, epochs=5, verbose=2)
# dump(decoder_model, open('model-ep{epoch:03d}-loss{loss:.3f}', 'wb'))


## to check if the model can work within a loop


epochs = 5
for i in range(epochs):
    generator = batch_generator(512, transfer_values_train, train_img_num, train_token)
    decoder_model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch, epochs=1, verbose=2)

'''for data generator'''
#
# epochs=5
# for i in range(epochs):
#     generator = data_generator(train_token_dict_marked, train_features, tokenizer, max_length,len(word_list))
#     decoder_model.fit_generator(generator, epochs=1, steps_per_epoch=300, verbose=2)




'''not using generator'''

# X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_token_dict_marked,
#                                             train_features,len(word_list))
#
# X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_token_dict_marked,
#                                          test_features,len(word_list))
#
#
# # define checkpoint callback
# filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# # fit model
# decoder_model.fit([X1train, X2train], ytrain, epochs=3, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))



# best_model = load_model('model-ep004-loss5.023-val_loss4.961.h5')


'''maybe something wrong with this function'''


def word_for_id(integer, tok):
	for word, index in tok.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tok, img_name, maxlength):
	# seed the generation process
	in_text = 'ssss'
	# iterate over the whole length of the sequence
	for i in range(maxlength):
		# integer encode input sequence
		sequence = tok.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=maxlength)
		# predict next word
        # photo = extract_feature(img_name)

		yhat = model.predict([extract_feature(img_name),sequence], verbose=0)
		# convert probability to integer
		yhat = np.argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tok)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'eeee':
			break

	return in_text


output_text = generate_desc(decoder_model,tokenizer,'Flickr8k/Flicker8k_Dataset/160585932_fa6339f248.jpg', max_length)


# all_features = load(open('img_features.pkl', 'rb'))
# test_feature = extract_feature(all_features, '56489627_e1de43de34.jpg')

print()
print('Predict caption:')
print(output_text)
print()

# print('True caption:')
# print(true_caption)
# print()



# evaluate the skill of the model
# def evaluate_model(test_dict, maxlength):
#     true, predict = [], []
#     sentence_score =0
#
#     for key, caption in test_dict.items():
#         y_pred, y_true = generate_captions(key,maxlength)
#
#         score = sentence_bleu(y_true, y_pred)
#         sentence_score = sentence_score + score
#
#         true.append(y_true)
#         predict.append(y_pred)
#
#     sentence_score = np.mean(sentence_score, dtype = np.float32)
#
#     print('BLEU-1: %f' % corpus_bleu(true, predict, weights=(1.0, 0, 0, 0)))
#     print('BLEU-2: %f' % corpus_bleu(true, predict, weights=(0.5, 0.5, 0, 0)))
#     print('BLEU-3: %f' % corpus_bleu(true, predict, weights=(0.3, 0.3, 0.3, 0)))
#     print('BLEU-4: %f' % corpus_bleu(true, predict, weights=(0.25, 0.25, 0.25, 0.25)))
#     print('BLEU-5: %f' % sentence_score)
#
#
# evaluate_model(test_token_dict_marked,max_length)



#
# def get_bleu_score(dataset):
#     score = 0
#
#     for img_name in dataset.splitlines():
#         predict_caption, true_caption = generate_captions(img_name)
#         score = sentence_bleu(true_caption, predict_caption)
#         score += score
#     score = np.mean(score, dtype=np.float32)
#
#     return score
#
# val_score = get_bleu_score(val_img_read)
# print('BLEU score of validation set: ', val_score)

