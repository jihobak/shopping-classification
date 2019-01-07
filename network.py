# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input, Conv1D, MaxPool1D, Flatten, BatchNormalization, GlobalMaxPool1D, ELU, concatenate, Conv2D, MaxPool2D, Add, ReLU, LeakyReLU, GlobalAveragePooling2D
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation

from keras import backend as K

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextImg_512D:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextImg_512d')

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            merge = concatenate([embd_out, img])
            
            num_b = 57
            num_m = 552
            num_s = 3190
            num_d = 404
            
            fc_d_cate = Dense(512, name='fc_d_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc_d_cate)
            fc_a = ELU()(fc_drop)
            
            d_cate = Dense(num_d, activation='softmax', name='d_cate')(fc_a)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[d_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    #'b_cate':'sparse_categorical_crossentropy'
                    #'m_cate':'sparse_categorical_crossentropy'
                    #'s_cate':'sparse_categorical_crossentropy'
                    'd_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    #'b_cate':'sparse_categorical_accuracy'
                    #'m_cate':'sparse_categorical_accuracy'
                    #'s_cate':'sparse_categorical_accuracy'
                    'd_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextImg_1024D:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextImg_1024d')

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            merge = concatenate([embd_out, img])
            
            num_b = 57
            num_m = 552
            num_s = 3190
            num_d = 404
            
            fc_d_cate = Dense(1024, name='fc_d_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc_d_cate)
            fc_a = ELU()(fc_drop)
            
            d_cate = Dense(num_d, activation='softmax', name='d_cate')(fc_a)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[d_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    #'b_cate':'sparse_categorical_crossentropy'
                    #'m_cate':'sparse_categorical_crossentropy'
                    #'s_cate':'sparse_categorical_crossentropy'
                    'd_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    #'b_cate':'sparse_categorical_accuracy'
                    #'m_cate':'sparse_categorical_accuracy'
                    #'s_cate':'sparse_categorical_accuracy'
                    'd_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

    
class TextImg_1024_512S:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextImg_1024_512s')

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            merge = concatenate([embd_out, img])
            
            num_b = 57
            num_m = 552
            num_s = 3190
            num_d = 404
            
            fc_s_cate = Dense(1024, name='fc_s_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc_s_cate)
            fc_a = ELU()(fc_drop)
            
            fc_s_cate2 = Dense(512, name='fc_s_cate2')(fc_a)
            fc_drop2 = Dropout(rate=0.5)(fc_s_cate2)
            fc_a2 = ELU()(fc_drop2)
            
            s_cate = Dense(num_s, activation='softmax', name='s_cate')(fc_a2)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[s_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    #'b_cate':'sparse_categorical_crossentropy'
                    #'m_cate':'sparse_categorical_crossentropy'
                    's_cate':'sparse_categorical_crossentropy'
                    #'d_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    #'b_cate':'sparse_categorical_accuracy'
                    #'m_cate':'sparse_categorical_accuracy'
                    's_cate':'sparse_categorical_accuracy'
                    #'d_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextImg_1024S:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextImg_1024s')

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            merge = concatenate([embd_out, img])
            
            num_b = 57
            num_m = 552
            num_s = 3190
            num_d = 404
            
            fc_s_cate = Dense(1024, name='fc_s_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc_s_cate)
            fc_a = ELU()(fc_drop)
            
            s_cate = Dense(num_s, activation='softmax', name='s_cate')(fc_a)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[s_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    #'b_cate':'sparse_categorical_crossentropy'
                    #'m_cate':'sparse_categorical_crossentropy'
                    's_cate':'sparse_categorical_crossentropy'
                    #'d_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    #'b_cate':'sparse_categorical_accuracy'
                    #'m_cate':'sparse_categorical_accuracy'
                    's_cate':'sparse_categorical_accuracy'
                    #'d_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

    
class TextImg_1024B:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextImg_1024b')

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            merge = concatenate([embd_out, img])
            
            num_b = 57
            num_m = 552
            num_s = 3190
            num_d = 404
            
            fc_b_cate = Dense(1024, name='fc_b_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc_b_cate)
            fc_a = ELU()(fc_drop)
            
            b_cate = Dense(num_b, activation='softmax', name='b_cate')(fc_a)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[b_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    'b_cate':'sparse_categorical_crossentropy'
                    #'m_cate':'sparse_categorical_crossentropy'
                    #'s_cate':'sparse_categorical_crossentropy'
                    #'d_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    'b_cate':'sparse_categorical_accuracy'
                    #'m_cate':'sparse_categorical_accuracy'
                    #'s_cate':'sparse_categorical_accuracy'
                    #'d_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextImg_512B:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextImg_512b')

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            merge = concatenate([embd_out, img])
            
            num_b = 57
            num_m = 552
            num_s = 3190
            num_d = 404
            
            fc_b_cate = Dense(512, name='fc_b_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc_b_cate)
            fc_a = ELU()(fc_drop)
            
            b_cate = Dense(num_b, activation='softmax', name='b_cate')(fc_a)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[b_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    'b_cate':'sparse_categorical_crossentropy'
                    #'m_cate':'sparse_categorical_crossentropy'
                    #'s_cate':'sparse_categorical_crossentropy'
                    #'d_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    'b_cate':'sparse_categorical_accuracy'
                    #'m_cate':'sparse_categorical_accuracy'
                    #'s_cate':'sparse_categorical_accuracy'
                    #'d_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextConv1:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextConv1')
    
    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
        
            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)
            """
            desn1 = Dense(256)(line)
            a4 = ELU()(desn1)
            bn4 = BatchNormalization()(a4)

            desn2 = Dense(128)(bn4)
            a5 = ELU()(desn2)
            bn5 = BatchNormalization()(a5)
            """
            num_m = 552
            
            outputs = Dense(num_m, activation='softmax')(line)

            model = Model(inputs=[t_uni], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optm,
                          metrics=['sparse_categorical_accuracy'])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
            
class TextConv1_512:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextConv1_512')
    
    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
        
            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)
            
            desn = Dense(512)(line)
            a4 = ELU()(desn)
            bn4 = BatchNormalization()(a4)
            
            num_m = 552
            
            outputs = Dense(num_m, activation='softmax')(bn4)

            model = Model(inputs=[t_uni], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optm,
                          metrics=['sparse_categorical_accuracy'])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

class TextConv1_1024:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextConv1_1024')
    
    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
        
            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)

            desn = Dense(1024)(line)
            a4 = ELU()(desn)
            bn4 = BatchNormalization()(a4)
            
            num_m = 552
            
            outputs = Dense(num_m, activation='softmax')(bn4)

            model = Model(inputs=[t_uni], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optm,
                          metrics=['sparse_categorical_accuracy'])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

class TextConv1_256_128:
    """
    """
    def __init__(self):
        self.logger = get_logger('TextConv1_256_128')
    
    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
        
            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)
            
            desn1 = Dense(256)(line)
            a4 = ELU()(desn1)
            bn4 = BatchNormalization()(a4)

            desn2 = Dense(128)(bn4)
            a5 = ELU()(desn2)
            bn5 = BatchNormalization()(a5)
            
            num_m = 552
            
            outputs = Dense(num_m, activation='softmax')(bn5)

            model = Model(inputs=[t_uni], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optm,
                          metrics=['sparse_categorical_accuracy'])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
            
class TextConCat:
    """
    uni_embd = tuni* w_uni
    uni_embd와 tuni의 concat을 활용
    """
    def __init__(self):
        self.logger = get_logger('TextConCat')
    
    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
        
            w_uni = Input((max_len,), name="input_2") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1) #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)

            merge = concatenate([line, uni_embd])
            
            num_m = 552

            outputs = Dense(num_m, activation='softmax')(merge)

            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)

            model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=optm,
                        metrics=['sparse_categorical_accuracy'])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
            
class TextConCat_1024:
    """
    uni_embd = tuni* w_uni
    uni_embd와 tuni의 concat을 활용
    """
    def __init__(self):
        self.logger = get_logger('TextConCat_1024')
    
    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
        
            w_uni = Input((max_len,), name="input_2") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1) #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)

            merge = concatenate([line, uni_embd])

            desn1 = Dense(1024)(merge)
            a4 = ELU()(desn1)
            bn4 = BatchNormalization()(a4)
            
            num_m = 552

            outputs = Dense(num_m, activation='softmax')(bn4)

            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)

            model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=optm,
                        metrics=['sparse_categorical_accuracy'])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

    
class Text:
    def __init__(self, logger_name, cate, fc_hidden=1024):
        self.cate = cate
        self.fc_hidden = fc_hidden
        self.logger = get_logger(logger_name)
        
    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        
        # Gpu config setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))
        
        
        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)
            
            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = ReLU(name='embd_out')(embd_out_pre)
            
            #add = Add()([text_output, img_output])
            
            """
            conv1 = Conv1D(128, 3)(t_uni_embd)
            bn1 = BatchNormalization()(conv1)
            a1 = ReLU()(bn1)
            maxp1 = MaxPool1D(3)(a1)

            conv2 = Conv1D(256, 3)(maxp1)
            bn2 = BatchNormalization()(conv2)
            a2 = ReLU()(bn2)

            conv3 = Conv1D(512, 3)(a2)
            bn3 = BatchNormalization()(conv3)
            a3 = ReLU()(bn3)

            line = GlobalMaxPool1D()(a3)
            """

            #merge = concatenate([line, uni_embd])
            
            
            if self.cate == 'b':
                num_classes = 57
            elif self.cate == 'm':
                num_classes = 552
            elif self.cate == 's':
                num_classes = 3190
            else:
                num_classes = 404
            
            fc = Dense(self.fc_hidden, name='fc_'+self.cate+'_cate')(embd_out)
            fc_drop = Dropout(rate=0.5)(fc)
            fc_a = ReLU()(fc_drop)
            
            output_name = self.cate+'_cate'
            cate = Dense(num_classes, activation='softmax', name=output_name)(fc_a)
            
            model = Model(inputs=[t_uni, w_uni], outputs=[cate]) #
            optm = keras.optimizers.Adam(opt.lr) 
            model.compile(
                loss={
                    output_name:'sparse_categorical_crossentropy'
                },
                optimizer=optm,
                metrics={
                    output_name:'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        
        return model    
    

class TextImg:
    """
    onlym_textimg_1024.weights.22-0.75.hdf5
    valid accuracy: 0.84
    Now, this is the best model.
    """
    def __init__(self, logger_name, cate, fc_hidden=1024):
        self.cate = cate
        self.fc_hidden = fc_hidden
        self.logger = get_logger(logger_name)#('TextImg_'+str(fc_hidden)+self.cate)

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        # Gpu config setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            #embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            embd_out = ReLU(name='embd_out')(embd_out_pre)
            
            img = Input((2048,), name="image")
            """
            img_reshaped = Reshape((45, 45, 1))(img)
            img_conv1 = Conv2D(128, 5)(img_reshaped)
            img_a1 = ELU()(img_conv1)
            img_maxp1 = MaxPool2D()(img_a1) 
            img_bn1 = BatchNormalization()(img_maxp1)

            img_conv2 = Conv2D(256, 5)(img_bn1)
            img_a2 = ELU()(img_conv2)
            img_maxp2 = MaxPool2D()(img_a2)
            img_bn2 = BatchNormalization()(img_maxp2)

            img_conv3 = Conv2D(256, 3)(img_bn2)
            img_a3 = ELU()(img_conv3)
            img_maxp3 = MaxPool2D()(img_a3)
            img_bn3 = BatchNormalization()(img_maxp3)

            img_flat = Flatten()(img_bn3)
            """
            merge = concatenate([embd_out, img])
            
            if self.cate == 'b':
                num_classes = 57
            elif self.cate == 'm':
                num_classes = 552
            elif self.cate == 's':
                num_classes = 3190
            else:
                num_classes = 404
            
            #num_b = 57
            #num_m = 552
            #num_s = 3190
            #num_d = 404
            
            #merge_drop = Dropout(rate=0.5)(merge)
            fc = Dense(self.fc_hidden, name='fc_'+self.cate+'_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc)
            fc_a = ReLU()(fc_drop)
            
            
            #fc_m_cate2 = Dense(512, name='fc2_m_cate')(fc_bn)
            #fc_a2 = ELU()(fc_m_cate2)
            #fc_bn2 = BatchNormalization()(fc_a2)
            
            output_name = self.cate+'_cate'
            
            output = Dense(num_classes, activation='softmax', name=output_name)(fc_a)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[output]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    #'b_cate':'sparse_categorical_crossentropy'
                    output_name:'sparse_categorical_crossentropy'
                    #'s_cate':'sparse_categorical_crossentropy'
                    #'d_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    #'b_cate':'sparse_categorical_accuracy'
                    output_name:'sparse_categorical_accuracy'
                    #'s_cate':'sparse_categorical_accuracy'
                    #'d_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

    
class TextImg2Layer:
    """
    onlym_textimg_1024.weights.22-0.75.hdf5
    valid accuracy: 0.84
    Now, this is the best model.
    """
    def __init__(self, cate, fc_hidden1=1024, fc_hidden2=512):
        self.cate = cate
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.logger = get_logger('TextImg_'+str(fc_hidden1)+'_'+str(fc_hidden2)+self.cate)

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        # Gpu config setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            """
            img_reshaped = Reshape((45, 45, 1))(img)
            img_conv1 = Conv2D(128, 5)(img_reshaped)
            img_a1 = ELU()(img_conv1)
            img_maxp1 = MaxPool2D()(img_a1) 
            img_bn1 = BatchNormalization()(img_maxp1)

            img_conv2 = Conv2D(256, 5)(img_bn1)
            img_a2 = ELU()(img_conv2)
            img_maxp2 = MaxPool2D()(img_a2)
            img_bn2 = BatchNormalization()(img_maxp2)

            img_conv3 = Conv2D(256, 3)(img_bn2)
            img_a3 = ELU()(img_conv3)
            img_maxp3 = MaxPool2D()(img_a3)
            img_bn3 = BatchNormalization()(img_maxp3)

            img_flat = Flatten()(img_bn3)
            """
            merge = concatenate([embd_out, img])
            
            if self.cate == 'b':
                num_classes = 57
            elif self.cate == 'm':
                num_classes = 552
            elif self.cate == 's':
                num_classes = 3190
            else:
                num_classes = 404
            
            #num_b = 57
            #num_m = 552
            #num_s = 3190
            #num_d = 404
            
            fc_1_cate = Dense(self.fc_hidden1, name='fc_'+self.cate+'_cate1')(merge)
            fc_drop = Dropout(rate=0.5)(fc_1_cate)
            fc_a = ELU()(fc_drop)
            
            fc_2_cate = Dense(self.fc_hidden2, name='fc_'+self.cate+'_cate2')(fc_a)
            fc_drop2 = Dropout(rate=0.5)(fc_2_cate)
            fc_a2 = ELU()(fc_drop2)
            
            
            #fc_m_cate2 = Dense(512, name='fc2_m_cate')(fc_bn)
            #fc_a2 = ELU()(fc_m_cate2)
            #fc_bn2 = BatchNormalization()(fc_a2)
            
            output_name = self.cate+'_cate'
            
            m_cate = Dense(num_classes, activation='softmax', name=output_name)(fc_a2)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[m_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    #'b_cate':'sparse_categorical_crossentropy'
                    output_name:'sparse_categorical_crossentropy'
                    #'s_cate':'sparse_categorical_crossentropy'
                    #'d_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    #'b_cate':'sparse_categorical_accuracy'
                    output_name:'sparse_categorical_accuracy'
                    #'s_cate':'sparse_categorical_accuracy'
                    #'d_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
    

    
class TextImgDual_1024:
    """
    """
    def __init__(self, cate):
        self.cate = cate
        self.logger = get_logger('TextImg_1024'+self.cate)

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        # Gpu config setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            
            merge = concatenate([embd_out, img])
            
            if self.cate == 'b':
                num_classes = 57
            elif self.cate == 'm':
                num_classes = 552
            elif self.cate == 's':
                num_classes = 3190
            elif self.cate == 'bm':
                num_classes_b = 57
                num_classes_m = 552
            else:
                num_classes = 404
            
            
            fc_m_cate = Dense(1024, name='fc_b_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc_m_cate)
            fc_a = ELU()(fc_drop)
            
            
            fc_b_cate = Dense(1024, name='fc_m_cate')(merge)
            fc_drop2 = Dropout(rate=0.5)(fc_b_cate)
            fc_a2 = ELU()(fc_drop2)
            
            
            b_cate = Dense(num_classes_b, activation='softmax', name='b_cate')(fc_a)
            m_cate = Dense(num_classes_m, activation='softmax', name='m_cate')(fc_a2)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[b_cate, m_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    'b_cate':'sparse_categorical_crossentropy',
                    'm_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    'b_cate':'sparse_categorical_accuracy',
                    'm_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

    
class TextImgDualLossWeight_1024:
    """
    """
    def __init__(self, cate):
        self.cate = cate
        self.logger = get_logger('TextImg_1024'+self.cate)

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        # Gpu config setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            
            merge = concatenate([embd_out, img])
            
            if self.cate == 'b':
                num_classes = 57
            elif self.cate == 'm':
                num_classes = 552
            elif self.cate == 's':
                num_classes = 3190
            elif self.cate == 'bm':
                num_classes_b = 57
                num_classes_m = 552
            else:
                num_classes = 404
            
            
            fc_m_cate = Dense(1024, name='fc_b_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc_m_cate)
            fc_a = ELU()(fc_drop)
            
            
            fc_b_cate = Dense(1024, name='fc_m_cate')(merge)
            fc_drop2 = Dropout(rate=0.5)(fc_b_cate)
            fc_a2 = ELU()(fc_drop2)
            
            
            b_cate = Dense(num_classes_b, activation='softmax', name='b_cate')(fc_a)
            m_cate = Dense(num_classes_m, activation='softmax', name='m_cate')(fc_a2)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[b_cate, m_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    'b_cate':'sparse_categorical_crossentropy',
                    'm_cate':'sparse_categorical_crossentropy'
                },
                loss_weights={
                    'b_cate':1.1,
                    'm_cate':1.2
                },
                optimizer=optm,
                metrics={
                    'b_cate':'sparse_categorical_accuracy',
                    'm_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

                                 
class TextAddImg:
    """
    Input: text, img
    text에서 나온 softmax 결과와 
    img에서 나온 softmax 결과를 더 한결과를 다시 
    final softmax에 넣어보자
    """
    def __init__(self):
        self.logger = get_logger('TextAddImg')

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            
            img = Input((2048,), name="image")
            
            num_m = 552
            
            text_output = Dense(num_m, activation='softmax',name='text_output')(embd_out)
            img_output = Dense(num_m, activation='softmax', name='img_output')(img)
            
            add = Add()([text_output, img_output])
            
            m_cate = Dense(num_m, activation='softmax', name='m_cate')(add)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[m_cate])
            optm = keras.optimizers.Adam(opt.lr)
            model.compile(
                loss={
                    'm_cate':'sparse_categorical_crossentropy'
                },
                optimizer=optm,
                metrics={
                    'm_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
            

class Img:
    """
    """
    def __init__(self, logger_name, cate, fc_hidden=1024, base=False):
        self.cate = cate
        self.fc_hidden = fc_hidden
        self.logger = get_logger(logger_name)
        self.base = base

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        # Gpu config setting
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))


        with tf.device('/gpu:0'):
            
            img = Input((opt.img_2dsize,), name="image")
            
            if self.base:
                img_flat = img
            else:
                """
                img_reshaped = Reshape((45, 45, 1))(img)
                img_conv1 = Conv2D(128, 5)(img_reshaped)
                img_bn1 = BatchNormalization()(img_conv1)
                img_a1 = ReLU()(img_bn1)
                img_maxp1 = MaxPool2D()(img_a1) 


                img_conv2 = Conv2D(256, 3)(img_maxp1)
                img_bn2 = BatchNormalization()(img_conv2)
                img_a2 = ReLU()(img_bn2)
                img_maxp2 = MaxPool2D()(img_a2)


                img_conv3 = Conv2D(512, 3)(img_maxp2)
                img_bn3 = BatchNormalization()(img_conv3)
                img_a3 = ReLU()(img_bn3)
                img_maxp3 = MaxPool2D()(img_a3)


                img_flat = GlobalAveragePooling2D()(img_maxp3)
                """
                img_flat = img
            
            
            if self.cate == 'b':
                num_classes = 57
            elif self.cate == 'm':
                num_classes = 552
            elif self.cate == 's':
                num_classes = 3190
            else:
                num_classes = 404
            
            
            #merge_drop = Dropout(rate=0.5)(merge)
            fc = Dense(self.fc_hidden, name='fc_'+self.cate+'_cate')(img_flat)
            fc_drop = Dropout(rate=0.5)(fc)
            fc_a = ReLU()(fc_drop)
            
            output_name = self.cate+'_cate'
            output = Dense(num_classes, activation='softmax', name=output_name)(fc_a)
           
            model = Model(inputs=[img], outputs=[output]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    output_name:'sparse_categorical_crossentropy'
                },
                optimizer=optm,
                metrics={
                    output_name:'sparse_categorical_accuracy'
                }
            )
            model.summary(print_fn=lambda x: self.logger.info(x))
            
        return model


class AdvTextOnly:
    def __init__(self):
        self.logger = get_logger('AdvTextOnly')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            """
            img = Input((2025,), name="image")
            img_reshaped = Reshape((45, 45, 1))(img)
            img_conv1 = Conv2D(128, 5)(img_reshaped)
            img_a1 = ELU()(img_conv1)
            img_maxp1 = MaxPool2D()(img_a1) 
            img_bn1 = BatchNormalization()(img_maxp1)

            img_conv2 = Conv2D(256, 5)(img_bn1)
            img_a2 = ELU()(img_conv2)
            img_maxp2 = MaxPool2D()(img_a2)
            img_bn2 = BatchNormalization()(img_maxp2)

            img_conv3 = Conv2D(256, 3)(img_bn2)
            img_a3 = ELU()(img_conv3)
            img_maxp3 = MaxPool2D()(img_a3)
            img_bn3 = BatchNormalization()(img_maxp3)

            img_flat = Flatten()(img_bn3)
            
            merge = concatenate([embd_out, img_flat])
            """
            num_b = 57
            num_m = 552
            num_s = 3190
            num_d = 404
            
            #merge_b_1 = Dense(256)(merge)
            #merge_b_1a = ELU()(merge_b_1)
            #merge_b_2 = Dense(128)(merge_b_1a)
            #merge_b_2a = ELU()(merge_b_2)
            
            #merge_m_1 = Dense(512)(merge)
            #merge_m_1a = ELU()(merge_m_1)
            #merge_m_2 = Dense(256)(merge_m_1a)
            #merge_m_2a = ELU()(merge_m_2)
            
            #merge_s = Dense(512)(merge)
            #merge_s_a = ELU()(merge_s)
            #merge_s2 = Dense(256)(merge_s_a)
            #merge_s2_a = ELU()(merge_s2)
            
            #merge_d = Dense(256)(merge)
            #merge_d_a = ELU()(merge_d)
            
            #b_cate = Dense(num_b, activation='softmax', name='b_cate')(embd_out)
            fc_m_cate = Dense(256, name='fc_m_cate')(embd_out)
            fc_drop = Dropout(rate=0.5)(fc_m_cate)
            fc_a = ELU()(fc_drop)
            m_cate = Dense(num_m, activation='softmax', name='m_cate')(fc_a)
            #s_cate = Dense(num_s, activation='softmax', name='s_cate')(embd_out)
            #d_cate = Dense(num_d, activation='softmax', name='d_cate')(embd_out)
            
            
            #outputs = Dense(num_classes, activation='softmax')(merge) #'softmax' activation
            model = Model(inputs=[t_uni, w_uni], outputs=[m_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    #'b_cate':'sparse_categorical_crossentropy'
                    'm_cate':'sparse_categorical_crossentropy'
                    #'s_cate':'sparse_categorical_crossentropy'
                    #'d_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    #'b_cate':'sparse_categorical_accuracy'
                    'm_cate':'sparse_categorical_accuracy'
                    #'s_cate':'sparse_categorical_accuracy'
                    #'d_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model

    
class TextOnly_1024:
    def __init__(self):
        self.logger = get_logger('textonly_1024')

    def get_model(self):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            num_m = 552
            
            fc_m_cate = Dense(1024, name='fc_m_cate')(embd_out)
            fc_drop = Dropout(rate=0.5)(fc_m_cate)
            fc_a = ELU()(fc_drop)
            m_cate = Dense(num_m, activation='softmax', name='m_cate')(fc_a)
            
            model = Model(inputs=[t_uni, w_uni], outputs=[m_cate]) #
            optm = keras.optimizers.Adam(opt.lr) 
            model.compile(
                loss={
                    'm_cate':'sparse_categorical_crossentropy'
                },
                optimizer=optm,
                metrics={
                    'm_cate':'sparse_categorical_accuracy'
                }
            )
            
            model.summary(print_fn=lambda x: self.logger.info(x))
        
        return model


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="word_input") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)
            
            w_uni = Input((max_len,), name="wweight_input") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1, name="weightd_word_matrix") #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out_pre = Dropout(rate=0.5)(uni_embd)
            embd_out = Activation('relu', name='embd_out')(embd_out_pre)
            
            """
            img = Input((2025,), name="image")
            img_reshaped = Reshape((45, 45, 1))(img)
            img_conv1 = Conv2D(128, 5)(img_reshaped)
            img_a1 = ELU()(img_conv1)
            img_maxp1 = MaxPool2D()(img_a1) 
            img_bn1 = BatchNormalization()(img_maxp1)

            img_conv2 = Conv2D(256, 5)(img_bn1)
            img_a2 = ELU()(img_conv2)
            img_maxp2 = MaxPool2D()(img_a2)
            img_bn2 = BatchNormalization()(img_maxp2)

            img_conv3 = Conv2D(256, 3)(img_bn2)
            img_a3 = ELU()(img_conv3)
            img_maxp3 = MaxPool2D()(img_a3)
            img_bn3 = BatchNormalization()(img_maxp3)

            img_flat = Flatten()(img_bn3)
            
            merge = concatenate([embd_out, img_flat])
            """
            num_b = 57
            num_m = 552
            num_s = 3190
            num_d = 404
            
            #merge_b_1 = Dense(256)(merge)
            #merge_b_1a = ELU()(merge_b_1)
            #merge_b_2 = Dense(128)(merge_b_1a)
            #merge_b_2a = ELU()(merge_b_2)
            
            #merge_m_1 = Dense(512)(merge)
            #merge_m_1a = ELU()(merge_m_1)
            #merge_m_2 = Dense(256)(merge_m_1a)
            #merge_m_2a = ELU()(merge_m_2)
            
            #merge_s = Dense(512)(merge)
            #merge_s_a = ELU()(merge_s)
            #merge_s2 = Dense(256)(merge_s_a)
            #merge_s2_a = ELU()(merge_s2)
            
            #merge_d = Dense(256)(merge)
            #merge_d_a = ELU()(merge_d)
            
            #b_cate = Dense(num_b, activation='softmax', name='b_cate')(embd_out)
            #m_cate = Dense(num_m, activation='softmax', name='m_cate')(embd_out)
            s_cate = Dense(num_s, activation='softmax', name='s_cate')(embd_out)
            #d_cate = Dense(num_d, activation='softmax', name='d_cate')(embd_out)
            
            
            #outputs = Dense(num_classes, activation='softmax')(merge) #'softmax' activation
            model = Model(inputs=[t_uni, w_uni], outputs=[s_cate]) # t_uni, w_uni, img
            optm = keras.optimizers.Adam(opt.lr) # Adam Nadam
            model.compile(
                loss={
                    #'b_cate':'sparse_categorical_crossentropy'
                    #'m_cate':'sparse_categorical_crossentropy'
                    's_cate':'sparse_categorical_crossentropy'
                    #'d_cate':'sparse_categorical_crossentropy'
                },
                #loss_weights={
                #    'b_cate':1.1,
                #    'm_cate':1.2
                    #'s_cate':1.3,
                    #'d_cate':1.4
                #},
                optimizer=optm,
                metrics={
                    #'b_cate':'sparse_categorical_accuracy'
                    #'m_cate':'sparse_categorical_accuracy'
                    's_cate':'sparse_categorical_accuracy'
                    #'d_cate':'sparse_categorical_accuracy'
                }
            )
            
            """
            img = Input((2048,), name="image")
            img_reshaped = Reshape((64, 32, 1))(img)
            img_conv1 = Conv2D(128, 3)(img_reshaped)
            img_a1 = ELU()(img_conv1)
            img_maxp1 = MaxPool2D()(img_a1) 
            img_bn1 = BatchNormalization()(img_maxp1)

            img_conv2 = Conv2D(256, 3)(img_bn1)
            img_a2 = ELU()(img_conv2)
            img_maxp2 = MaxPool2D()(img_a2)
            img_bn2 = BatchNormalization()(img_maxp2)

            img_conv3 = Conv2D(256, 3)(img_bn2)
            img_a3 = ELU()(img_conv3)
            img_maxp3 = MaxPool2D()(img_a3)
            img_bn3 = BatchNormalization()(img_maxp3)

            img_flat = Flatten()(img_bn3)
            
            w_uni = Input((max_len,), name="input_2") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1) #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)

            merge = concatenate([line, uni_embd, img_flat])

            desn1 = Dense(256)(merge)
            a4 = ELU()(desn1)
            bn4 = BatchNormalization()(a4)

            desn2 = Dense(128)(bn4)
            a5 = ELU()(desn2)
            bn5 = BatchNormalization()(a5)

            outputs = Dense(num_classes, activation='softmax')(bn5)

            model = Model(inputs=[t_uni, w_uni, img], outputs=outputs)
            optm = keras.optimizers.Nadam(1e-4)

            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            """
            
            """
            4th
            
            w_uni = Input((max_len,), name="input_2") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1) #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)

            merge = concatenate([line, uni_embd])

            desn1 = Dense(256)(merge)
            a4 = ELU()(desn1)
            bn4 = BatchNormalization()(a4)

            desn2 = Dense(128)(bn4)
            a5 = ELU()(desn2)
            bn5 = BatchNormalization()(a5)

            outputs = Dense(num_classes, activation='softmax')(bn5)

            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(1e-4)

            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            """

            """
            3th

            conv1 = Conv1D(128, 5)(t_uni_embd)
            a1 = ELU()(conv1)
            bn1 = BatchNormalization()(a1)

            conv2 = Conv1D(256, 5)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 5)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)


            conv4 = Conv1D(512, 5)(bn3)
            a4 = ELU()(conv4)
            bn4 = BatchNormalization()(a4)


            conv5 = Conv1D(512, 3)(bn4)
            a5 = ELU()(conv5)
            bn5 = BatchNormalization()(a5)


            conv6 = Conv1D(512, 3)(bn5)
            a6 = ELU()(conv6)
            bn6 = BatchNormalization()(a6)


            conv7 = Conv1D(128, 3)(bn6)
            a7 = ELU()(conv7)
            maxp = MaxPool1D(3)(a7)
            bn7 = BatchNormalization()(maxp)


            line = Flatten()(bn7)

            desn1 = Dense(256)(line)
            a8 = ELU()(desn1)
            bn8 = BatchNormalization()(a8)

            desn2 = Dense(128)(bn8)
            a9 = ELU()(desn2)
            bn9 = BatchNormalization()(a9)

            outputs = Dense(num_classes, activation='softmax')(bn9)
            
            model = Model(inputs=[t_uni], outputs=outputs)
            optm = keras.optimizers.Adam(1e-4)

            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            """
            
            """
            2-th

            conv1 = Conv1D(128, 3)(t_uni_embd)
            a1 = ELU()(conv1)
            maxp1 = MaxPool1D(3)(a1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3)(bn1)
            a2 = ELU()(conv2)
            bn2 = BatchNormalization()(a2)


            conv3 = Conv1D(512, 3)(bn2)
            a3 = ELU()(conv3)
            bn3 = BatchNormalization()(a3)

            line = GlobalMaxPool1D()(bn3)

            desn1 = Dense(256)(line)
            a4 = ELU()(desn1)
            bn4 = BatchNormalization()(a4)

            desn2 = Dense(128)(bn4)
            a5 = ELU()(desn2)
            bn5 = BatchNormalization()(a5)

            outputs = Dense(num_classes, activation='softmax')(bn5)

            model = Model(inputs=[t_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(1e-4)

            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            """
            
            """
            첫번째 모델
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')
            
            t_uni = Input((max_len,), name="input_1") # (None, 32)
            t_uni_embd = embd(t_uni)  # token, (None, 32, 128)

            conv1 = Conv1D(128, 3, activation='relu')(t_uni_embd)

            maxp1 = MaxPool1D(3)(conv1)
            bn1 = BatchNormalization()(maxp1)

            conv2 = Conv1D(256, 3, activation='relu')(bn1)
            maxp2 = MaxPool1D(3)(conv2)
            bn2 = BatchNormalization()(maxp2)

            line = Flatten()(bn2)

            desn1 = Dense(256, activation='relu')(line)
            drop1 = Dropout(rate=0.5)(desn1)
            bn3 = BatchNormalization()(drop1)

            desn2 = Dense(128, activation='relu')(bn3)
            drop2 = Dropout(rate=0.5)(desn2)
            bn4 = BatchNormalization()(drop2)

            outputs = Dense(num_classes, activation='softmax')(bn4)

            model = Model(inputs=[t_uni], outputs=outputs)
            optm = keras.optimizers.Adam(1e-4)

            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            """

            """
            베이스라인
            
            w_uni = Input((max_len,), name="input_2") # (None, 32)
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight (None, 32, 1)

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1) #(None, 32, 128) dot (None, 32, 1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat) #(None, 128)

            embd_out = Dropout(rate=0.5)(uni_embd)
            relu = Activation('relu', name='relu1')(embd_out)
            outputs = Dense(num_classes, activation=activation)(relu)
            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            """
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
