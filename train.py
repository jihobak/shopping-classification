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

import os
import json
import threading

import fire
import h5py
import tqdm
import numpy as np
import six

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from six.moves import zip, cPickle

from misc import get_logger, Option
from network import TextImg
opt = Option('./config.json')
if six.PY2:
    cate1 = json.loads(open('../cate1.json').read())
else:
    cate1 = json.loads(open('../cate1.json', 'rb').read().decode('utf-8'))

DEV_DATA_LIST = opt.dev_data_list
TEST_DATA_LIST = opt.test_data_list

class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0
    
    def get_textimg_generator(self, ds, batch_size, cate, size, raise_stop_event=False):
        left, limit = 0, size
        
        if cate == 'b':
            cate_index = 'bindex'
        elif cate == 'm':
            cate_index = 'mindex'
        elif cate == 's':
            cate_index = 'sindex'
        else:
            cate_index = 'dindex'    
        
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
            
            Y = [ds[t][left:right] for t in [cate_index]] ###
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
    
    def train_textimg(self, data_root, data_file_name, out_dir, cate, fc_hidden):
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')

        output_dir_base = "only"+cate+"_khaiii2_textimg_"+str(fc_hidden)

        self.weight_fname = os.path.join(out_dir, output_dir_base+".weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        
        train = data['train']
        dev = data['dev']
        
        callbacks_list = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max',
                period=10
            ),
            CSVLogger(output_dir_base+'_log.csv',
                append=True,
                separator=',')
        ]
        
        
        # b or m
        num_size = 6503449
        num_size_valid = 1625910
        
        # s
        num_ssize = 5012525
        num_ssize_valid = 1252930
        
        # d
        num_dsize = 605398
        num_dsize_valid = 151714
        
        
        # Train
        if cate == 'b':
            total_train_samples = num_size
            total_dev_samples = num_size_valid
        elif cate == 'm':
            total_train_samples = num_size
            total_dev_samples = num_size_valid
        elif cate == 's':
            total_train_samples = num_ssize
            total_dev_samples = num_ssize_valid
        else:
            total_train_samples = num_dsize
            total_dev_samples = num_dsize_valid
        
        
        textimg = TextImg(output_dir_base, cate, fc_hidden)
        textimg_model = textimg.get_model()
        
        train_gen = self.get_textimg_generator(
                                            train,
                                            opt.batch_size,
                                            cate,
                                            total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        dev_gen = self.get_textimg_generator(
                                          dev,
                                          opt.batch_size,
                                          cate,
                                          total_dev_samples
                                          )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))
        
        textimg_model.fit_generator(generator=train_gen,
                        steps_per_epoch=self.steps_per_epoch,
                        epochs=opt.num_epochs,
                        validation_data=dev_gen,
                        validation_steps=self.validation_steps,
                        shuffle=True,
                        callbacks=callbacks_list)


if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({
               'train_textimg': clsf.train_textimg
              })
