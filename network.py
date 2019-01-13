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
from keras.layers import Dense, Input, concatenate, ReLU
from keras.layers.core import Reshape
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation
from keras import backend as K

from misc import get_logger, Option
opt = Option('./config.json')


class TextImg:
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
            
            img = Input((2048,), name="image")
    
            merge = concatenate([embd_out, img])
            
            if self.cate == 'b':
                num_classes = 57
            elif self.cate == 'm':
                num_classes = 552
            elif self.cate == 's':
                num_classes = 3190
            else:
                num_classes = 404
            
            fc = Dense(self.fc_hidden, name='fc_'+self.cate+'_cate')(merge)
            fc_drop = Dropout(rate=0.5)(fc)
            fc_a = ReLU()(fc_drop)
            

            output_name = self.cate+'_cate'
            
            output = Dense(num_classes, activation='softmax', name=output_name)(fc_a)
            
            model = Model(inputs=[t_uni, w_uni, img], outputs=[output])
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
