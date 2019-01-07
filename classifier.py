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
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, TensorBoard
from six.moves import zip, cPickle

from misc import get_logger, Option
from network import TextOnly, top1_acc, AdvTextOnly, Img, Text, TextImg, TextImg2Layer, TextConv1, TextConv1_512, TextConv1_1024, TextConv1_256_128, \
                    TextConCat, TextConCat_1024, TextAddImg, TextOnly_1024, TextImg_512B, TextImg_1024B, TextImg_1024S, TextImg_1024D, TextImg_512D, TextImg_1024_512S, TextImgDual_1024, TextImgDualLossWeight_1024

opt = Option('./config.json')
if six.PY2:
    cate1 = json.loads(open('../cate1.json').read())
else:
    cate1 = json.loads(open('../cate1.json', 'rb').read().decode('utf-8'))
DEV_DATA_LIST = opt.dev_data_list#dev_data_list#test_data_list #['../dev.chunk.01']


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0

    def get_sample_generator(self, ds, batch_size, raise_stop_event=False):
        left, limit = 0, ds['uni'].shape[0]
        #left, limit = 0, index
        
        while True:
            right = min(left + batch_size, limit)
            #X = [ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
            X = [ds[t][left:right, :] for t in ['uni', 'w_uni']]
            #X = [ds[t][left:right, :] for t in ['uni']]
            #X = [ds[t][left:right, :2025]for t in ['img']]
            #X = [ds[t][left:right, :2025] if t == 'img' else ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
            #Y = ds['cate'][left:right]

            # 4항목 모두 트레이닝 시킬때
            #Y = [ds[t][left:right] for t in ['bindex', 'mindex', 'sindex', 'dindex']]

            # 2항목 b, m 만 트레이닝 시킬때
            #Y = [ds[t][left:right] for t in ['bindex', 'mindex']]
            
            # b 만 트레이닝 시킬떄
            Y = [ds[t][left:right] for t in ['mindex']] ###
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
    
    def get_text_generator(self, ds, batch_size, cate, size, raise_stop_event=False):
        #left, limit = 0, ds['uni'].shape[0]
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
            X = [ds[t][left:right, :] for t in ['uni', 'w_uni']]
            Y = [ds[t][left:right] for t in [cate_index]] ###
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
                    
    def get_img_generator(self, ds, batch_size, cate, size, raise_stop_event=False):
        #left, limit = 0, ds['uni'].shape[0]
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
            
            X = [ds[t][left:right, :opt.img_2dsize]for t in ['img']]
            Y = [ds[t][left:right] for t in [cate_index]] ###
            
            yield X, Y
            
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
    
    def get_textimg_generator(self, ds, batch_size, cate, size, raise_stop_event=False):
        #left, limit = 0, ds['uni'].shape[0]
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
    
    def get_textimg_bmgenerator(self, ds, batch_size, index, raise_stop_event=False):
        #left, limit = 0, ds['uni'].shape[0]
        left, limit = 0, index
        
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] if t == 'img' else ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
            
            # b, m 만 트레이닝 시킬떄
            Y = [ds[t][left:right] for t in ['bindex', 'mindex']] ###
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
    
                    
    def get_textimg_bgenerator(self, ds, batch_size, index, raise_stop_event=False):
        #left, limit = 0, ds['uni'].shape[0]
        left, limit = 0, index
        
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] if t == 'img' else ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
            
            # m 만 트레이닝 시킬떄
            Y = [ds[t][left:right] for t in ['bindex']] ###
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
    
    def get_textimg_sgenerator(self, ds, batch_size, index, raise_stop_event=False):
        #left, limit = 0, ds['uni'].shape[0]
        left, limit = 0, index
        
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] if t == 'img' else ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
            
            # m 만 트레이닝 시킬떄
            Y = [ds[t][left:right] for t in ['sindex']] ###
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
                    
    
    def get_textimg_dgenerator(self, ds, batch_size, index, raise_stop_event=False):
        #left, limit = 0, ds['uni'].shape[0]
        left, limit = 0, index
        
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] if t == 'img' else ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
            
            # m 만 트레이닝 시킬떄
            Y = [ds[t][left:right] for t in ['dindex']] ###
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
                    
                    
                    
    def get_uni_generator(self, ds, batch_size, cate, size, raise_stop_event=False):
        #left, limit = 0, ds['uni'].shape[0]
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
            X = [ds[t][left:right, :] for t in ['uni']]
            Y = [ds[t][left:right] for t in [cate_index]] ###
            
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration
                    
    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for d in ['b', 'm', 's', 'd']:
            inv_cate1[d] = {v: k for k, v in six.iteritems(cate1[d])}
        return inv_cate1

    def write_prediction_result(self, data, pred_y, meta, out_path, readable):
        pid_order = []
        for data_path in DEV_DATA_LIST:
            h = h5py.File(data_path, 'r')['dev']
            pid_order.extend(h['pid'][::])
        
        #pid_order.extend(data['pid'][::])
        
        b_vocab, m_vocab, s_vocab, d_vocab = meta
        
        by2l = {i: s for s, i in six.iteritems(b_vocab)}
        by2l = list(map(lambda x: x[1], sorted(by2l.items(), key=lambda x: x[0])))
        
        my2l = {i: s for s, i in six.iteritems(m_vocab)}
        my2l = list(map(lambda x: x[1], sorted(my2l.items(), key=lambda x: x[0])))
        
        sy2l = {i: s for s, i in six.iteritems(s_vocab)}
        sy2l = list(map(lambda x: x[1], sorted(sy2l.items(), key=lambda x: x[0])))
        
        dy2l = {i: s for s, i in six.iteritems(d_vocab)}
        dy2l = list(map(lambda x: x[1], sorted(dy2l.items(), key=lambda x: x[0])))
                
        #y2l = {i: s for s, i in six.iteritems(meta['y_vocab'])}
        #y2l = list(map(lambda x: x[1], sorted(y2l.items(), key=lambda x: x[0])))
        
        inv_cate1 = self.get_inverted_cate1(cate1)
        
        rets = {}
        
        """
        pred_y = [(b, m, s, d),
        (b, m, s, d),
        (b, m, s, d),
        (b, m, s, d),
        ....
        ]
        """
        for pid, y in zip(data['pid'], pred_y):
            if six.PY3:
                pid = pid.decode('utf-8')
            #label = y2l[y]
            #tkns = list(map(int, label.split('>')))
            #b, m, s, d = tkns
            b = by2l[y[0]]
            m = my2l[y[1]]
            s = sy2l[y[2]]
            d = dy2l[y[3]]
            
            assert b in inv_cate1['b']
            assert m in inv_cate1['m']
            assert s in inv_cate1['s']
            assert d in inv_cate1['d']
            tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'
            if readable:
                b = inv_cate1['b'][b]
                m = inv_cate1['m'][m]
                s = inv_cate1['s'][s]
                d = inv_cate1['d'][d]
            rets[pid] = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                if six.PY3:
                    pid = pid.decode('utf-8')
                ans = rets.get(pid, no_answer.format(pid=pid))
                fout.write(ans)
                fout.write('\n')

    def predict(self, data_root, model_root, test_root, test_div, out_path, readable=False):
        #meta_path = os.path.join(data_root, 'meta')
        #meta = cPickle.loads(open(meta_path, 'rb').read())
        
        b_vocab = cPickle.loads(open("./data/b_vocab.cPickle", 'rb').read())
        m_vocab = cPickle.loads(open("./data/m_vocab.cPickle", 'rb').read())
        s_vocab = cPickle.loads(open("./data/s_vocab.cPickle", 'rb').read())
        d_vocab = cPickle.loads(open("./data/d_vocab.cPickle", 'rb').read())
        
        meta = (b_vocab, m_vocab, s_vocab, d_vocab)
        
        #model_fname = os.path.join(model_root, 'model.h5')
        #model_fname = os.path.join(model_root, 'onlys_withword.weights.82-1.04.hdf5') ###
        
        ##############
        #최종 예측할때 ##
        ##############
        #bmodel_fname = os.path.join(model_root, 'onlyb_withword.weights.26-0.54.hdf5')
        #mmodel_fname = os.path.join(model_root, 'onlym_withword.weights.49-0.91.hdf5')# m2
        #mmodel_fname = os.path.join(model_root, 'onlym_textimg_1024.weights.22-0.75.hdf5') #m1 
        #smodel_fname = os.path.join(model_root, 'onlys_withword.weights.82-1.04.hdf5')
        #dmodel_fname = os.path.join(model_root, 'onlyd_withword.weights.76-0.53.hdf5')
        
        mmodel_fname = os.path.join(model_root, 'onlym_textimg_1024.weights.22-0.75.hdf5')
        
        self.logger.info('bcate # of classes(train): %s' % len(b_vocab))
        self.logger.info('mcate # of classes(train): %s' % len(m_vocab))
        self.logger.info('scate # of classes(train): %s' % len(s_vocab))
        self.logger.info('dcate # of classes(train): %s' % len(d_vocab))
        
        #model = load_model(model_fname,
        #                   custom_objects={'top1_acc': top1_acc})
        
        #bmodel = load_model(bmodel_fname)
        mmodel = load_model(mmodel_fname)
        #smodel = load_model(smodel_fname)
        #dmodel = load_model(dmodel_fname)
        
        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        batch_size = opt.batch_size
        pred_y = []
        test_gen = ThreadsafeIter(self.get_textimg_generator(test, batch_size, raise_stop_event=True))
        total_test_samples = test['uni'].shape[0]
        
        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:
                total_test_samples = test['uni'].shape[0]
                X, _ = chunk
                #_pred_y = model.predict(X)
                
                #_pred_b = bmodel.predict(X)
                _pred_m = mmodel.predict(X)
                #_pred_s = smodel.predict(X)
                #_pred_d = dmodel.predict(X)
                
                
                #banswer = [np.argmax(p, axis=-1) for p in _pred_b]
                manswer = [np.argmax(p, axis=-1) for p in _pred_m]
                #sanswer = [np.argmax(p, axis=-1) for p in _pred_s]
                #danswer = [np.argmax(p, axis=-1) for p in _pred_d]
                
                #answer = [np.argmax(p, axis=-1) for p in _pred_y]
                #answer = [(a, b, c, d) for a, b, c, d in zip(*answer)]
                #answer = [(a, 0, 0, 0) for a, b in zip(*answer)]
                #answer = [(0, 0, a, 0) for a in answer]
                
                #answer = [(b, m, s, d) for b, m, s, d in zip(banswer, manswer, sanswer, danswer)]
                answer = [(0, a, 0, 0) for a in manswer]
                pred_y.extend(answer)
                pbar.update(X[0].shape[0])
                
        self.write_prediction_result(test, pred_y, meta, out_path, readable=readable)

    
    def predict_m(self, data_root, model_root, test_root, test_div, out_path, data_filename, cate, readable=False):
        """
        python classifier.py predict_m ./data/train ./model/train ./data/train/ dev onlym_khaiii_textimg1024_predict.tsv khaiii_data.h5py m
        """
        b_vocab = cPickle.loads(open("./data/b_vocab.cPickle", 'rb').read())
        m_vocab = cPickle.loads(open("./data/m_vocab.cPickle", 'rb').read())
        s_vocab = cPickle.loads(open("./data/s_vocab.cPickle", 'rb').read())
        d_vocab = cPickle.loads(open("./data/d_vocab.cPickle", 'rb').read())
        
        meta = (b_vocab, m_vocab, s_vocab, d_vocab)
        
        ##############
        #최종 예측할때 ##
        ##############
        mmodel_fname = os.path.join(model_root, 'onlym_khaiii_textimg1024.weights.81-0.83.hdf5') #best, m3
        #mmodel_fname = os.path.join(model_root, 'onlym_withword.weights.49-0.91.hdf5')# m2
        #mmodel_fname = os.path.join(model_root, 'onlym_textimg_1024.weights.22-0.75.hdf5') #m1 
        
        self.logger.info('mcate # of classes(train): %s' % len(m_vocab))

        mmodel = load_model(mmodel_fname)
        
        test_path = os.path.join(test_root, data_filename)
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        batch_size = opt.batch_size
        pred_y = []
        test_gen = ThreadsafeIter(self.get_textimg_generator(test, batch_size, cate, raise_stop_event=True))
        total_test_samples = test['uni'].shape[0]
        
        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:
                total_test_samples = test['uni'].shape[0]
                X, _ = chunk
                _pred_m = mmodel.predict(X)
                
                manswer = [np.argmax(p, axis=-1) for p in _pred_m]
                answer = [(0, a, 0, 0) for a in manswer]
                pred_y.extend(answer)
                pbar.update(X[0].shape[0])
                
        self.write_prediction_result(test, pred_y, meta, out_path, readable=readable)
    
    
    def predict_all(self, data_root, model_root, test_root, test_div, out_path, readable=False):
        """
        python classifier.py predict_all ./data/train ./model/train ./data/train/ dev valid_khaiii2_12_512textimgdrop_relu_cw_1024_predict.tsv
        
        python classifier.py predict_all ./data/train ./model/train/Final ./data/dev/ dev dev_fuck_predict.tsv
        
        python classifier.py predict_all ./data/train ./model/train/Final ./data/test/ dev dev_final_predict.tsv
        """
        b_vocab = cPickle.loads(open("./data/b_vocab.cPickle", 'rb').read())
        m_vocab = cPickle.loads(open("./data/m_vocab.cPickle", 'rb').read())
        s_vocab = cPickle.loads(open("./data/s_vocab.cPickle", 'rb').read())
        d_vocab = cPickle.loads(open("./data/d_vocab.cPickle", 'rb').read())
        
        meta = (b_vocab, m_vocab, s_vocab, d_vocab)
        
        
        ##############
        #최종 예측할때 ##
        ##############
        #bmodel_fname = os.path.join(model_root, 'onlyb_textimg_512.weights.38-0.90.hdf5')
        #mmodel_fname = os.path.join(model_root, 'onlym_textimg_1024.weights.22-0.75.hdf5')# m2
        #smodel_fname = os.path.join(model_root, 'onlys_textimg_1024.weights.70-0.82.hdf5')
        #dmodel_fname = os.path.join(model_root, 'onlyd_textimg_512.weights.53-0.88.hdf5')
        
        # Khaiii textimg 1024
        #bmodel_fname = os.path.join(model_root, 'onlyb2_khaiii_textimg1024.weights.90-0.89.hdf5')
        #bmodel_fname = os.path.join(model_root, 'onlyb_khaiii_textimg1024.weights.70-0.89.hdf5')
        #mmodel_fname = os.path.join(model_root, 'onlym_khaiii_textimg1024.weights.81-0.83.hdf5')# m2
        #smodel_fname = os.path.join(model_root, 'onlys_khaiii_textimg1024.weights.40-0.80.hdf5')
        #dmodel_fname = os.path.join(model_root, 'onlyd_khaiii_textimg1024.weights.60-0.89.hdf5')
        
        # Khaiii2 textimg 1024
        #onlyb_khaiii2_textimg1024.weights.50-0.90
        bmodel_fname = os.path.join(model_root, 'onlyb_khaiii2_textimg1024.weights.50-0.90.hdf5')
        mmodel_fname = os.path.join(model_root, 'onlym_khaiii2_textimg1024.weights.60-0.84.hdf5')# m2
        smodel_fname = os.path.join(model_root, 'onlys_khaiii2_textimg1024.weights.70-0.82.hdf5')
        dmodel_fname = os.path.join(model_root, 'onlyd_khaiii2_textimg1024.weights.50-0.89.hdf5')
        
        
        # Khaiii2_12_512textimgdrop_relu_cw_1024
        # drop 을 적용시킨 것과 안 적용 시킨것 두개다 제출 해볼 것임
        # drop
        #bmodel_fname = os.path.join(model_root, 'onlyb_khaiii2_12_512textimgdrop_relu_cw_1024.weights.20-0.90.hdf5')
        #mmodel_fname = os.path.join(model_root, 'onlym_khaiii2_12_512textimg_1024.weights.30-0.85.hdf5')# m2
        #smodel_fname = os.path.join(model_root, 'onlys_khaiii2_12_512textimgdrop_relu_cw_1024.weights.60-0.83.hdf5')
        #dmodel_fname = os.path.join(model_root, 'onlyd_khaiii2_12_512textimgdrop_relu_cw_1024.weights.30-0.89.hdf5')
        # no drop
        #bmodel_fname = os.path.join(model_root, 'onlyb_khaiii2_12_512textimg_relu_cw_1024.weights.20-0.90.hdf5')
        #mmodel_fname = os.path.join(model_root, 'onlym_khaiii2_12_512textimg_1024.weights.30-0.85.hdf5')# m2
        #smodel_fname = os.path.join(model_root, 'onlys_khaiii2_12_512textimgdrop_relu_cw_1024.weights.60-0.83.hdf5')
        #dmodel_fname = os.path.join(model_root, 'onlyd_khaiii2_12_512textimg_relu_cw_1024.weights.30-0.89.hdf5')
        
        
        
        self.logger.info('bcate # of classes(train): %s' % len(b_vocab))
        self.logger.info('mcate # of classes(train): %s' % len(m_vocab))
        self.logger.info('scate # of classes(train): %s' % len(s_vocab))
        self.logger.info('dcate # of classes(train): %s' % len(d_vocab))
        
        
        bmodel = load_model(bmodel_fname)
        mmodel = load_model(mmodel_fname)
        smodel = load_model(smodel_fname)
        dmodel = load_model(dmodel_fname)
                                             
        test_path = os.path.join(test_root, 'dev_khaiii2_data.h5py') #### dev_khaiii2new_data_120000 , khaiii2_data_120000
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        batch_size = opt.batch_size
        
        pred_y = []
        
        total_test_samples = test['uni'].shape[0]
        test_gen = ThreadsafeIter(self.get_textimg_generator(test, batch_size, 'a', total_test_samples, raise_stop_event=True))
        
        
        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:
                #total_test_samples = test['uni'].shape[0]
                X, _ = chunk
                
                _pred_b = bmodel.predict(X)
                _pred_m = mmodel.predict(X)
                _pred_s = smodel.predict(X)
                _pred_d = dmodel.predict(X)
                
                
                banswer = [np.argmax(p, axis=-1) for p in _pred_b]
                manswer = [np.argmax(p, axis=-1) for p in _pred_m]
                sanswer = [np.argmax(p, axis=-1) for p in _pred_s]
                danswer = [np.argmax(p, axis=-1) for p in _pred_d]
                
                
                answer = [(b, m, s, d) for b, m, s, d in zip(banswer, manswer, sanswer, danswer)]
                pred_y.extend(answer)
                pbar.update(X[0].shape[0])
                
        self.write_prediction_result(test, pred_y, meta, out_path, readable=readable)
    
    
    def train_text(self, data_root, data_file_name, out_dir, cate, fc_hidden, load_class_weight):
        """
        #python classifier.py train_piper ./data/train khaiii2_data.h5py ./model/train m 1024
        
        python classifier.py train_text ./data/train khaiii2_data_120000_bmd.h5py ./model/train d 1024 True
        """
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        
        if load_class_weight:
            output_dir_base = "only"+cate+"_khaiii2_12_512text_baseline_relu_cw_"+str(fc_hidden)
        else:
            output_dir_base = "only"+cate+"_khaiii2_12_512text_baseline_relu_"+str(fc_hidden)
            
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
            CSVLogger(output_dir_base+'_log.csv', ##
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
        
        
        text = Text(output_dir_base, cate, fc_hidden)
        text_model = text.get_model()
        
        train_gen = self.get_text_generator(
                                            train,
                                            opt.batch_size,
                                            cate,
                                            total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        dev_gen = self.get_text_generator(
                                          dev,
                                          opt.batch_size,
                                          cate,
                                          total_dev_samples
                                          )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        # Load class weight
        if load_class_weight:
            weight_file_name = './model/train/'+cate+'class_weights.npy'
            
            class_weight = np.load(weight_file_name, 'r')

            text_model.fit_generator(generator=train_gen,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=opt.num_epochs,
                                validation_data=dev_gen,
                                validation_steps=self.validation_steps,
                                shuffle=True,
                                class_weight = class_weight,
                                callbacks=callbacks_list)
        else:
            text_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
        
    
    def train_textimg(self, data_root, data_file_name, out_dir, cate, fc_hidden, load_class_weight):
        """
        python classifier.py train_textimg ./data/train khaiii2_data_120000.h5py ./model/train b 1024 True
        python classifier.py train_textimg ./data/train khaiii2_data_120000.h5py ./model/train m 1024 True
        
        khaiii2_data_tfidf_12000
        
        python classifier.py train_textimg ./data/train khaiii2_data_120000_bms.h5py ./model/train s 1024 True
        
        python classifier.py train_textimg ./data/train khaiii2_data_120000_bmd.h5py ./model/train d 1024 True
        
        python classifier.py train_textimg ./data/train khaiii2_data20_120000_bmd.h5py ./model/train d 1024 True
        
        python classifier.py train_textimg ./data/train khaiii2_data_tfidf_12000.h5py ./model/train b 1024 True
        """
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        
        if load_class_weight:
            output_dir_base = "only"+cate+"_khaiii2_tfidf_12_512textimgdrop_relu_cw_"+str(fc_hidden)
        else:
            output_dir_base = "only"+cate+"_khaiii2_tfidf_12_512textimgdrop_relu_"+str(fc_hidden)
            
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
            CSVLogger(output_dir_base+'_log.csv', ##
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
        
        # Load class weight
        if load_class_weight:
            weight_file_name = './model/train/'+cate+'class_weights.npy'
            
            class_weight = np.load(weight_file_name, 'r')

            textimg_model.fit_generator(generator=train_gen,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=opt.num_epochs,
                                validation_data=dev_gen,
                                validation_steps=self.validation_steps,
                                shuffle=True,
                                class_weight = class_weight,
                                callbacks=callbacks_list)
        else:
            textimg_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
    
    def train_img(self, data_root, data_file_name, out_dir, cate, fc_hidden, load_class_weight, base):
        """
        python classifier.py train_img ./data/train khaiii2_data_120000.h5py ./model/train b 256 True True
        """
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        
        if load_class_weight:
            if base:
                output_dir_base = "only"+cate+"_khaiii2_12_512imgbase_relu_cw_"+str(fc_hidden)
            else:
                output_dir_base = "only"+cate+"_khaiii2_12_512img_relu_cw_"+str(fc_hidden)
        else:
            if base:
                output_dir_base = "only"+cate+"_khaiii2_12_512imgbase_relu_"+str(fc_hidden)
            else:
                output_dir_base = "only"+cate+"_khaiii2_12_512img_relu_"+str(fc_hidden)
            
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
            CSVLogger(output_dir_base+'_log.csv', ##
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
        
        
        img = Img(output_dir_base, cate, fc_hidden)
        img_model = img.get_model()
        
        train_gen = self.get_img_generator(
                                            train,
                                            opt.batch_size,
                                            cate,
                                            total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        dev_gen = self.get_img_generator(
                                          dev,
                                          opt.batch_size,
                                          cate,
                                          total_dev_samples
                                          )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))
        
        # Load class weight
        if load_class_weight:
            weight_file_name = './model/train/'+cate+'class_weights.npy'
            
            class_weight = np.load(weight_file_name, 'r')

            img_model.fit_generator(generator=train_gen,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=opt.num_epochs,
                                validation_data=dev_gen,
                                validation_steps=self.validation_steps,
                                shuffle=True,
                                class_weight = class_weight,
                                callbacks=callbacks_list)
        else:
            img_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
            
        
    def train_m(self, data_root, data_file_name, out_dir):
        """
        python classifier.py train_m ./data/train khaiii2_data.h5py ./model/train
        python classifier.py train_m ./data/train khaiii2_data_120000.h5py ./model/train
        """
        print("start")
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        self.weight_fname = os.path.join(out_dir, "onlym_khaiii2_12_512textimg_1024.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
        
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
            CSVLogger('onlym_khaiii2_12_512textimg_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        num_size = 6503449
        num_size_valid = 1625910
        
        textimg_m = TextImg('m', 1024)
        textimg_m_model = textimg_m.get_model()
        
        total_train_samples = num_size
        # get_textimg_generator
        train_gen = self.get_textimg_generator(
                                            train,
                                            opt.batch_size,
                                            'm',
                                            total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_size_valid
        dev_gen = self.get_textimg_generator(
                                             dev,
                                             opt.batch_size,
                                             'm',
                                             total_dev_samples
                                             )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_m_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
    
    
    def train_b(self, data_root, data_file_name, out_dir):
        """
        python classifier.py train_b ./data/train khaiii2_data.h5py ./model/train
        """
        print("start")
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        self.weight_fname = os.path.join(out_dir, "onlyb_khaiii2_textimg512.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
        
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
            CSVLogger('onlyb_khaiii2_textimg_512_log.csv', ##
                append=True,
                separator=',')
        ]
        
        num_size = 6503449
        num_size_valid = 1625910
        
        textimg_512b = TextImg('b', 512)
        textimg_512b_model = textimg_512b.get_model()
        
        total_train_samples = num_size
        train_gen = self.get_textimg_generator(
                                                train,
                                                opt.batch_size,
                                                'b',
                                                total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_size_valid
        dev_gen = self.get_textimg_generator(
                                             dev,
                                             opt.batch_size,
                                             'b',
                                             total_dev_samples
                                             )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_512b_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
    
    def train_s(self, data_root, data_file_name, out_dir):
        """
        python classifier.py train_s ./data/train khaiii2_data_bms.h5py ./model/train
        """
        print("start")
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        self.weight_fname = os.path.join(out_dir, "onlys_khaiii2_textimg1024_512.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
        
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
            CSVLogger('onlys_khaiii2_textimg_1024_512_log.csv', ##
                append=True,
                separator=',')
        ]
        
        
        num_ssize = 5012525
        num_ssize_valid = 1252930
        
        textimg_s = TextImg2Layer('s', 1024, 512)
        textimg_s_model = textimg_s.get_model()
        
        total_train_samples = num_ssize
        train_gen = self.get_textimg_generator(
                                                train,
                                                opt.batch_size,
                                                's',
                                                total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_ssize_valid
        dev_gen = self.get_textimg_generator(
                                             dev,
                                             opt.batch_size,
                                             's',
                                             total_dev_samples
                                             )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_s_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
    
    
    def train_d(self, data_root, data_file_name, out_dir):
        """
        python classifier.py train_d ./data/train khaiii2_data_bmd.h5py ./model/train
        """
        print("start")
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        self.weight_fname = os.path.join(out_dir, "onlyd_khaiii2_textimg5124.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
        
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
            CSVLogger('onlyd_khaiii2_textimg_512_log.csv', ##
                append=True,
                separator=',')
        ]
        
        
        num_dsize = 605398
        num_dsize_valid = 151714
        
        textimg_512d = TextImg('d', 512)
        textimg_512d_model = textimg_512d.get_model()
        
        total_train_samples = num_dsize
        train_gen = self.get_textimg_generator(
                                                train,
                                                opt.batch_size,
                                                'd',
                                                total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_dsize_valid
        dev_gen = self.get_textimg_generator(
                                             dev,
                                             opt.batch_size,
                                             'd',
                                             total_dev_samples
                                             )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_512d_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
    
    def train_conv1(self, data_root, data_file_name, out_dir):
        """
        python classifier.py train_conv1 ./data/train khaiii2_data.h5py ./model/train
        """
        print("start")
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        self.weight_fname = os.path.join(out_dir, "onlym_khaiii2_textconv1.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
        
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
            CSVLogger('onlym_khaiii2_textconv1_log.csv', ##
                append=True,
                separator=',')
        ]
        
        
        num_size = 6503449
        num_size_valid = 1625910
        
        textconv1_m = TextConv1()
        textconv1_m_model = textconv1_m.get_model()
        
                          
        total_train_samples = num_size
        train_gen = self.get_uni_generator(
                                            train,
                                            opt.batch_size,
                                            'm',
                                            total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_size_valid
        dev_gen = self.get_uni_generator(
                                         dev,
                                         opt.batch_size,
                                         'm',
                                         total_dev_samples
                                        )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textconv1_m_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
    
    
    def train_dual(self, data_root, data_file_name, out_dir):
        """
        python classifier.py train_dual ./data/train khaiii_data.h5py ./model/train
        """
        print("start")
        data_path = os.path.join(data_root, data_file_name)
        data = h5py.File(data_path, 'r')
        self.weight_fname = os.path.join(out_dir, "onlybm_khaiii_textimg1024.weights.{epoch:02d}-{val_m_cate_sparse_categorical_accuracy:.2f}.hdf5")
        self.weight_fname2 = os.path.join(out_dir, "onlybm_with_weight_khaiii_textimg1024.weights.{epoch:02d}-{val_m_cate_sparse_categorical_accuracy:.2f}.hdf5")
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        
        train = data['train']
        dev = data['dev']
        
        callbacks_list = [
             EarlyStopping(
                monitor='val_m_cate_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname,
                monitor='val_m_cate_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max',
                period=5
            ),
            CSVLogger('onlybm_khaiii_textimg_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimgdual_1024 = TextImgDual_1024('bm')
        textimgdual_1024_model = textimgdual_1024.get_model()
        
        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_textimg_bmgenerator(
                                                train,
                                                opt.batch_size,
                                                total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))
        
        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_textimg_bmgenerator(
                                             dev,
                                             opt.batch_size,
                                             total_dev_samples
                                             )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimgdual_1024_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
        
        del textimgdual_1024
        
        callbacks_list2 = [
             EarlyStopping(
                monitor='val_m_cate_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname2,
                monitor='val_m_cate_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max',
                period=5
            ),
            CSVLogger('onlybm_with_weight_khaiii_textimg_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimgdual_loss_weight_1024 = TextImgDualLossWeight_1024('bm')
        textimgdual_loss_weight_1024_model = textimgdual_loss_weight_1024.get_model()
        
        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_textimg_bmgenerator(
                                                train,
                                                opt.batch_size,
                                                total_train_samples
                                               )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_textimg_bmgenerator(
                                             dev,
                                             opt.batch_size,
                                             total_dev_samples
                                             )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimgdual_loss_weight_1024_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list2)
        
    
    
    def train_all(self, data_root, out_dir):
        """
        python classifier.py train_all ./data/train ./model/train
        """
        sindex_size = 4938860
        dindex_size = 598275
        valid_sindex_size = 1234336
        valid_dindex_size = 149900
        
        
        #for c, data in zip(['b', 's', 'd'], ['khaiii_data.h5py', 'khaiii_data_bms.h5py', 'khaiii_data_bmd.h5py']):
        for c, data in zip(['s', 'd'], ['khaiii_data_bms.h5py', 'khaiii_data_bmd.h5py']):
            data_path = os.path.join(data_root, data)
            data = h5py.File(data_path, 'r')
            
            self.weight_fname = os.path.join(out_dir, "only"+c+"_khaiii_textimg1024.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5")
        
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            train = data['train']
            dev = data['dev']

            callbacks_list = [
                 EarlyStopping(
                    monitor='val_sparse_categorical_accuracy',
                    patience=10,
                    mode='max'
                ),
                ModelCheckpoint(
                    self.weight_fname,
                    monitor='val_sparse_categorical_accuracy',
                    save_best_only=True,
                    mode='max',
                    period=10
                ),
                CSVLogger('only'+c+'_khaiii_textimg_1024_log.csv', ##
                    append=True,
                    separator=',')
            ]

            textimg_1024 = TextImg_1024(c)
            textimg_1024_model = textimg_1024.get_model()
            
            
            
            # Train
            if c == 'b':
                total_train_samples = train['uni'].shape[0]
            elif c == 'm':
                total_train_samples = train['uni'].shape[0]
            elif c == 's':
                total_train_samples = sindex_size
            else:
                total_train_samples = dindex_size
            
            train_gen = self.get_textimg_generator(
                                                    train,
                                                    opt.batch_size,
                                                    c,
                                                    total_train_samples
                                                   )##
            self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))
            
            # Dev
            if c == 'b':
                total_dev_samples = dev['uni'].shape[0]
            elif c == 'm':
                total_dev_samples = dev['uni'].shape[0]
            elif c == 's':
                total_dev_samples = valid_sindex_size
            else:
                total_dev_samples = valid_dindex_size
                
            
            dev_gen = self.get_textimg_generator(
                                                 dev,
                                                 opt.batch_size,
                                                 c,
                                                 total_dev_samples
                                                 )##
            self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

            textimg_1024_model.fit_generator(generator=train_gen,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=opt.num_epochs,
                                validation_data=dev_gen,
                                validation_steps=self.validation_steps,
                                shuffle=True,
                                callbacks=callbacks_list)
            data.close()

    
    def train(self, data_root, out_dir):
        data_path = os.path.join(data_root, 'data.h5py') ####
        data_paths = os.path.join(data_root, 'bms_data.h5py')
        data_pathd = os.path.join(data_root, 'bmsd_data.h5py')
        meta_path = os.path.join(data_root, 'meta')
        
        data = h5py.File(data_path, 'r')
        datas = h5py.File(data_paths, 'r')
        datad = h5py.File(data_pathd, 'r')
        
        meta = cPickle.loads(open(meta_path, 'rb').read())
        
        
        # TextImg_512B,
        # TextImg_1024B,
        # TextImg_1024S, 
        # TextImg_1024_512S
        # TextImg_1024D, 
        # TextImg_512D
        
        #num_dindex = 605995 
        #num_dindex_dev = 151862
        
        #num_sindex = 5014987
        #num_sindex_dev = 1253557
        
        self.weight_fname = os.path.join(out_dir, "onlyb_textimg_512.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5") ##
        self.weight_fname2 = os.path.join(out_dir, "onlyb_textimg_1024.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5") ##
        self.weight_fname3 = os.path.join(out_dir, "onlys_textimg_1024.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5") ##
        self.weight_fname4 = os.path.join(out_dir, "onlys_textimg_1024_512.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5") ##
        self.weight_fname5 = os.path.join(out_dir, "onlyd_textimg_1024.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5") ##
        self.weight_fname6 = os.path.join(out_dir, "onlyd_textimg_512.weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5") ##
        #self.weight_fname7 = os.path.join(out_dir, "onlym_text_only_1024.weights.{epoch:02d}-{val_loss:.2f}.hdf5") ##
        #self.weight_fname8 = os.path.join(out_dir, "onlym_text_add_img.weights.{epoch:02d}-{val_loss:.2f}.hdf5") ##
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.num_classes = len(meta['y_vocab'])

        train = data['train']
        dev = data['dev']
        
        trains = datas['train']
        devs = datas['dev']
        
        traind = datad['train']
        devd = datad['dev']

        #self.logger.info('# of train samples: %s' % train['img'].shape[0])
        #self.logger.info('# of dev samples: %s' % dev['img'].shape[0])
        
        
        #total_train_samples = train['uni'].shape[0]
        #total_dev_samples = dev['uni'].shape[0]
        #self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))
        #self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))
        
        
        # TextImg_512B,
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
                mode='max'
            ),
            CSVLogger('onlyb_textimg_512_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimg_512b = TextImg_512B()
        textimg_512b_model = textimg_512b.get_model()
        
        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_textimg_bgenerator(train,
                                           opt.batch_size,
                                           total_train_samples
                                              )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_textimg_bgenerator(dev,
                                         opt.batch_size,
                                         total_dev_samples
                                        )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_512b_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
        
        
        del textimg_512b
        
        
        
        # TextImg_1024B,
        
        
        callbacks_list2 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname2,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlyb_textimg_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimg_1024b = TextImg_1024B()
        textimg_1024b_model = textimg_1024b.get_model()
        
        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_textimg_bgenerator(train,
                                           opt.batch_size,
                                           total_train_samples
                                              )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_textimg_bgenerator(dev,
                                         opt.batch_size,
                                         total_dev_samples
                                        )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_1024b_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list2)
        
        
        del textimg_1024b
        
        
        # TextImg_1024S,
        
        num_sindex = 5014987
        num_sindex_dev = 1253557
        callbacks_list3 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname3,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlys_textimg_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimg_1024s = TextImg_1024S()
        textimg_1024s_model = textimg_1024s.get_model()
        
        total_train_samples = num_sindex
        train_gen = self.get_textimg_sgenerator(trains,
                                           opt.batch_size,
                                           total_train_samples
                                              )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_sindex_dev
        dev_gen = self.get_textimg_sgenerator(devs,
                                         opt.batch_size,
                                         total_dev_samples
                                        )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_1024s_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list3)
        
        
        del textimg_1024s
        
        
        
        # TextImg_1024_512S
        
        num_sindex = 5014987
        num_sindex_dev = 1253557
        
        callbacks_list4 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname4,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlys_textimg_1024_512_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimg_1024_512s = TextImg_1024_512S()
        textimg_1024_512s_model = textimg_1024_512s.get_model()
        
        total_train_samples = num_sindex
        train_gen = self.get_textimg_sgenerator(trains,
                                           opt.batch_size,
                                           total_train_samples
                                              )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_sindex_dev
        dev_gen = self.get_textimg_sgenerator(devs,
                                         opt.batch_size,
                                         total_dev_samples
                                        )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_1024_512s_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list4)
        
        
        del textimg_1024_512s
        
        
        # TextImg_1024D,
        num_dindex = 605995 
        num_dindex_dev = 151862
        
        callbacks_list5 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname5,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlyd_textimg_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimg_1024_d = TextImg_1024D()
        textimg_1024_d_model = textimg_1024_d.get_model()
        
        total_train_samples = num_dindex
        train_gen = self.get_textimg_dgenerator(traind,
                                           opt.batch_size,
                                           total_train_samples
                                              )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_dindex_dev
        dev_gen = self.get_textimg_dgenerator(devd,
                                         opt.batch_size,
                                         total_dev_samples
                                        )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_1024_d_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list5)
        
        
        del textimg_1024_d
        
        
        # TextImg_512D
        num_dindex = 605995 
        num_dindex_dev = 151862
        
        callbacks_list6 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname6,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlyd_textimg_512_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimg_512_d = TextImg_512D()
        textimg_512_d_model = textimg_512_d.get_model()
        
        total_train_samples = num_dindex
        train_gen = self.get_textimg_dgenerator(traind,
                                           opt.batch_size,
                                           total_train_samples
                                              )##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = num_dindex_dev
        dev_gen = self.get_textimg_dgenerator(devd,
                                         opt.batch_size,
                                         total_dev_samples
                                        )##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        textimg_512_d_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list6)
        
        
        del textimg_512_d
        
        
        
        
        """
        
        
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
                mode='max'
            ),
            CSVLogger('onlym_text_conv1_log.csv', ##
                append=True,
                separator=',')
        ]
        
        
        ###################check
        #### 1.TextConv1 ####
        ###################
        text_conv1 = TextConv1()
        text_conv1_model = text_conv1.get_model()
        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_uni_generator(train,
                                           opt.batch_size)##
        
        #self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_uni_generator(dev,
                                         opt.batch_size)##
        
        #self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        text_conv1_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
        
        
        del text_conv1
        
        
        ########################check
        #### 2.TextConv1_512 ####
        ########################
        
        callbacks_list2 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname2,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlym_text_conv1_512_log.csv', ##
                append=True,
                separator=',')
        ]
        
        text_conv1_512 = TextConv1_512()
        text_conv1_512_model = text_conv1_512.get_model()
        
        train_gen = self.get_uni_generator(train,
                                           opt.batch_size)##
        dev_gen = self.get_uni_generator(dev,
                                         opt.batch_size)##

        text_conv1_512_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list2)
        
        del text_conv1_512
        
        
        
        ########################check
        #### 3.TextConv1_1024 ####
        ########################
        
        callbacks_list3 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname3,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlym_text_conv1_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        text_conv1_1024 = TextConv1_1024()
        text_conv1_1024_model = text_conv1_1024.get_model()
        
        train_gen = self.get_uni_generator(train,
                                           opt.batch_size)##
        dev_gen = self.get_uni_generator(dev,
                                         opt.batch_size)##

        text_conv1_1024_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list3)
        
        
        del text_conv1_1024
        
        
        ###########################check
        #### 4.TextConv1_256_128 ####
        ###########################
        
        callbacks_list4 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname4,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlym_text_conv1_256_128_log.csv', ##
                append=True,
                separator=',')
        ]
        
        text_conv1_256_128 = TextConv1_256_128()
        text_conv1_256_128_model = text_conv1_256_128.get_model()
        
        train_gen = self.get_uni_generator(train,
                                           opt.batch_size)##
        dev_gen = self.get_uni_generator(dev,
                                         opt.batch_size)##

        text_conv1_256_128_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list4)
        
        del text_conv1_256_128
        
        
        ####################check
        #### 5.TextConCat ####
        ####################
        
        callbacks_list5 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname5,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlym_text_concat_log.csv', ##
                append=True,
                separator=',')
        ]
        
        text_concat = TextConCat()
        text_concat_model = text_concat.get_model()
        
        train_gen = self.get_text_generator(train,
                                           opt.batch_size)##
        dev_gen = self.get_text_generator(dev,
                                         opt.batch_size)##

        text_concat_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list5)
        
        
        del text_concat
        
        
        ########################check
        ####6.TextConCat1024 ####
        ########################
        
        callbacks_list6 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname6,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlym_text_concat_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        text_concat_1024 = TextConCat_1024()
        text_concat_1024_model = text_concat_1024.get_model()
        
        train_gen = self.get_text_generator(train,
                                           opt.batch_size)##
        dev_gen = self.get_text_generator(dev,
                                         opt.batch_size)##

        text_concat_1024_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list6)
        
        
        del text_concat_1024
        
        
        
        ########################check
        #### 7. TextOnly_1024 ####
        ########################
        
        callbacks_list7 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname7,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlym_text_only_1024_log.csv', ##
                append=True,
                separator=',')
        ]
        
        text_only_1024 = TextOnly_1024()
        text_only_1024_model = text_only_1024.get_model()
        
        train_gen = self.get_text_generator(train,
                                           opt.batch_size)##
        dev_gen = self.get_text_generator(dev,
                                         opt.batch_size)##

        text_only_1024_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list7)
        
        del text_only_1024
        
        
        
        ########################check
        #### 8. TextAddImg ####
        ########################
        
        callbacks_list8 = [
             EarlyStopping(
                monitor='val_sparse_categorical_accuracy',
                patience=5,
                mode='max'
            ),
            ModelCheckpoint(
                self.weight_fname8,
                monitor='val_sparse_categorical_accuracy',
                save_best_only=True,
                mode='max'
            ),
            CSVLogger('onlym_textaddimg__log.csv', ##
                append=True,
                separator=',')
        ]
        
        textaddimg = TextAddImg()
        textaddimg_model = textaddimg.get_model()
        
        train_gen = self.get_textimg_generator(train,
                                           opt.batch_size)##
        dev_gen = self.get_textimg_generator(dev,
                                         opt.batch_size)##

        textaddimg_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list8)
        """
        
        
        
        """
        #num_dindex = 605995 # train: 605995 dev: 151862 ##
        #num_dindex_valid = 151862 ##
        
        #num_sindex = 5014987
        #num_sindex_valid = 1253557
        
        #textonly = TextOnly()
        #model = textonly.get_model(self.num_classes)
        img = ImgOnly()
        img_model = img.get_model(self.num_classes)

        total_train_samples = train['uni'].shape[0]
        #total_train_samples = num_sindex ##
        train_gen = self.get_img_generator(train,
                                           opt.batch_size)##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        #total_dev_samples = num_sindex_valid  ##
        dev_gen = self.get_img_generator(dev,
                                         opt.batch_size)##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        img_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list)
        
        textimg = TextImg()
        textimg_model = textimg.get_model(self.num_classes)

        total_train_samples = train['uni'].shape[0]
        #total_train_samples = num_sindex ##
        train_gen2 = self.get_textimg_generator(train,
                                           opt.batch_size)##
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        #total_dev_samples = num_sindex_valid  ##
        dev_gen2 = self.get_textimg_generator(dev,
                                            opt.batch_size)##
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))
        
        callbacks_list2 = [
             EarlyStopping(
                monitor='val_loss',
                patience=5,
            ),
            ModelCheckpoint(
                self.weight_fname2,
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            CSVLogger('onlym_textimg2_log.csv', ##
                append=True,
                separator=',')
        ]
        
        textimg_model.fit_generator(generator=train_gen2,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen2,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=callbacks_list2)
        """


class ThreadsafeIter(object):
    def __init__(self, it):
        self._it = it
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._it)

    def next(self):
        with self._lock:
            return self._it.next()


if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({'train': clsf.train,
               'train_b': clsf.train_b,
               'train_m': clsf.train_m,
               'train_s': clsf.train_s,
               'train_d': clsf.train_d,
               'train_text': clsf.train_text,
               'train_textimg': clsf.train_textimg,
               'train_img': clsf.train_img,
               'train_conv1': clsf.train_conv1,
               'train_dual': clsf.train_dual,
               'train_all':clsf.train_all,
               'predict': clsf.predict,
               'predict_m': clsf.predict_m,
               'predict_all':clsf.predict_all
              })
