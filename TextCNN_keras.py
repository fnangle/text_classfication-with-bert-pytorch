from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# dataset,info=tfds.load("imdb_reviews",with_info=True,as_supervised=True)

(train_data,train_label),(test_data,test_label)=keras.datasets.imdb.load_data(num_words=10000)
maxl=400
# print(len(train_data))
# print(train_data[0])
word_index=keras.datasets.imdb.get_word_index()
word2id={k:(v+3) for k,v in word_index.items()}
word2id['<PAD>'] = 0
word2id['<START>'] = 1
word2id['<UNK>'] = 2
word2id['<UNUSED>'] = 3
id2word={v:k for k,v in word_index.items()}
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# # We decode the review; note that our indices were offset by 3
# # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)
def get_words(sent_ids):
    return ' '.join([id2word.get(i-3,"?") for i in sent_ids]) #找不到返回？

sent = get_words(train_data[0])
# print(sent)
# 句子末尾padding
train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=word2id['<PAD>'],padding='post',maxlen=maxl)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value=word2id['<PAD>'],padding='post',maxlen=maxl)
# print('len: ',len(train_data[0]),len(test_data[1]))

#构建模型

vocab_size=10000
# model=keras.Input(shape=())
input=layers.Input(shape=(maxl, ))
em=layers.Embedding(vocab_size+1,300,input_length=maxl)(input)
cnn1=layers.Conv1D(256,kernel_size=3,padding='same',strides=1,activation='relu',activity_regularizer='l2')(em)
# cnn1 = layers.MaxPooling1D(2,strides=2)(cnn1)
# cnn1 = layers.MaxPooling1D(2)(cnn1)
# drop1 = layers.Dropout(0.25)(cnn1)
cnn2 = layers.Conv1D(filters=256, kernel_size=4, padding='same', strides=1, activation='relu',activity_regularizer='l2')(em)
# cnn2 = layers.MaxPooling1D(2,strides=2)(cnn2)
# cnn2 = layers.MaxPooling1D(2)(cnn2)
# drop2 = layers.Dropout(0.25)(cnn2)
cnn3 = layers.Conv1D(256, kernel_size=5, padding='same', strides=1, activation='relu',activity_regularizer='l2')(em)
# cnn3 = layers.MaxPooling1D(2)(cnn3)
# drop3 = layers.Dropout(0.25)(cnn3)
# concat=layers.concatenate([drop1, drop2, drop3], axis=-1)
concat=layers.concatenate([cnn1,cnn2 ,cnn3 ], axis=-1)
maxpool=layers.GlobalMaxPooling1D()(concat)
flat = layers.Flatten()(maxpool)
dense=layers.Dropout(0.5)(flat)
dense=layers.Dense(64,activation='relu')(dense)
output=layers.Dense(2,activation='softmax')(dense)
model=tf.keras.models.Model(input,output)

one_hot_labels = keras.utils.to_categorical(train_label, num_classes=2) #转为one hot编码
adamOpti=tf.keras.optimizers.Adam(0.001)
model.compile(loss='binary_crossentropy', optimizer=adamOpti, metrics=['accuracy'])

#callbacks_list=[#tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=1),
                #tf.keras.callbacks.ModelCheckpoint(filepath="bestmodel.h5",monitor='val_loss',save_best_only=True),
#	             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=0, mode='auto',
#                                                epsilon=0.0001, cooldown=0, min_lr=0)
#             ]

his=model.fit(train_data, one_hot_labels,batch_size=32, epochs=4,validation_split=0.1)
model.save("model.h5")

test_one_hot_labels=keras.utils.to_categorical(test_label, num_classes=2)
# model=tf.keras.models.load_model('model.h5')
plt.plot(his.history['accuracy'])
plt.plot(his.history['loss'])
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()
test_loss, test_acc = model.evaluate(test_data, test_one_hot_labels)
print('\n',test_loss,test_acc)
