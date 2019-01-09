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

class Inference():
    def __init__(self):
        self.logger = get_logger('Inference')
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
                       
    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for d in ['b', 'm', 's', 'd']:
            inv_cate1[d] = {v: k for k, v in six.iteritems(cate1[d])}
        return inv_cate1

    def write_prediction_result(self, data, pred_y, meta, out_path, readable):
        pid_order = []
        for data_path in TEST_DATA_LIST:
            h = h5py.File(data_path, 'r')['test']
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
    
    def predict_all(self, data_root, model_root, test_root, test_div, out_path, readable=False):
        b_vocab = cPickle.loads(open("./data/b_vocab.cPickle", 'rb').read())
        m_vocab = cPickle.loads(open("./data/m_vocab.cPickle", 'rb').read())
        s_vocab = cPickle.loads(open("./data/s_vocab.cPickle", 'rb').read())
        d_vocab = cPickle.loads(open("./data/d_vocab.cPickle", 'rb').read())
        
        meta = (b_vocab, m_vocab, s_vocab, d_vocab)
        
        # Khaiii2 textimg 1024
        bmodel_fname = os.path.join(model_root, 'onlyb_khaiii2_textimg_1024.weights.50-0.90.hdf5')
        mmodel_fname = os.path.join(model_root, 'onlym_khaiii2_textimg_1024.weights.60-0.84.hdf5')
        smodel_fname = os.path.join(model_root, 'onlys_khaiii2_textimg_1024.weights.70-0.82.hdf5')
        dmodel_fname = os.path.join(model_root, 'onlyd_khaiii2_textimg_1024.weights.50-0.89.hdf5')
        
        self.logger.info('bcate # of classes(train): %s' % len(b_vocab))
        self.logger.info('mcate # of classes(train): %s' % len(m_vocab))
        self.logger.info('scate # of classes(train): %s' % len(s_vocab))
        self.logger.info('dcate # of classes(train): %s' % len(d_vocab))
        
        
        bmodel = load_model(bmodel_fname)
        mmodel = load_model(mmodel_fname)
        smodel = load_model(smodel_fname)
        dmodel = load_model(dmodel_fname)
                                             
        test_path = os.path.join(test_root, 'khaiii2_data.h5py')
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        batch_size = opt.batch_size
        
        pred_y = []
        
        total_test_samples = test['uni'].shape[0]
        test_gen = ThreadsafeIter(self.get_textimg_generator(test, batch_size, 'a', total_test_samples, raise_stop_event=True))
        
        
        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:

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
    inf = Inference()
    fire.Fire({
               'predict_all':inf.predict_all
              })
