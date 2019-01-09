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
os.environ['OMP_NUM_THREADS'] = '1'
import pickle
import re
import sys
import traceback
from collections import Counter
from multiprocessing import Pool

import tqdm
import fire
import h5py
import numpy as np
import mmh3
import six
from six.moves import cPickle
from khaiii import KhaiiiApi
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


from misc import get_logger, Option
opt = Option('./config.json')

re_eng = re.compile('[^a-zA-Z]') # english
re_han = re.compile('[^ ㄱ-ㅣ가-힣]+') # han gul

class Reader(object):
    def __init__(self, data_path_list, div, begin_offset, end_offset):
        self.div = div
        self.data_path_list = data_path_list
        self.begin_offset = begin_offset
        self.end_offset = end_offset

    def is_range(self, i):
        if self.begin_offset is not None and i < self.begin_offset:
            return False
        if self.end_offset is not None and self.end_offset <= i:
            return False
        return True

    def get_size(self):
        offset = 0
        count = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[self.div]['pid'].shape[0]
            if not self.begin_offset and not self.end_offset:
                offset += sz
                count += sz
                continue
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                count += 1
            offset += sz
        return count

    def get_class(self, h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return (b, m, s, d)

    def generate(self):
        offset = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')[self.div]
            sz = h['pid'].shape[0]
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                bmsd = self.get_class(h, i)
                yield h['pid'][i], bmsd, h, i, h['img_feat'][i]
            offset += sz

    def get_y_vocab(self, data_path):
        y_vocab = {}
        h = h5py.File(data_path, 'r')[self.div]
        sz = h['pid'].shape[0]
        for i in tqdm.tqdm(range(sz), mininterval=1):
            class_name = self.get_class(h, i)
            if class_name not in y_vocab:
                y_vocab[class_name] = len(y_vocab)
        return y_vocab


def preprocessing(data):
    try:
        cls, data_path_list, div, out_path, begin_offset, end_offset, cate, dev= data
        data = cls()
        data.load_y_vocab()
        data.preprocessing(data_path_list, div, begin_offset, end_offset, out_path, cate, dev)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


class Data:
    tmp_chunk_tpl = 'tmp/base.chunk.%s'

    def __init__(self):
        self.logger = get_logger('data')
        self.api = KhaiiiApi()

    def load_y_vocab(self):
        self.b_vocab = cPickle.loads(open('./data/b_vocab.cPickle', 'rb').read())
        self.m_vocab = cPickle.loads(open('./data/m_vocab.cPickle', 'rb').read())
        self.s_vocab = cPickle.loads(open('./data/s_vocab.cPickle', 'rb').read())
        self.d_vocab = cPickle.loads(open('./data/d_vocab.cPickle', 'rb').read())

    def _split_data(self, data_path_list, div, chunk_size):
        total = 0
        
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
            
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]
        return chunks

    def preprocessing(self, data_path_list, div, begin_offset, end_offset, out_path, cate, dev):
        self.div = div
        reader = Reader(data_path_list, div, begin_offset, end_offset)
        rets = []
        for pid, bmsd, h, i, img in reader.generate():
            y, x = self.parse_data(bmsd, h, i, cate, dev)
            if y is None:
                continue
            
            # y = (b, m, s d)
            rets.append((pid, y, x, img))
        self.logger.info('sz=%s' % (len(rets)))
        open(out_path, 'wb').write(cPickle.dumps(rets, 2))
        self.logger.info('%s ~ %s done. (size: %s)' % (begin_offset, end_offset, end_offset - begin_offset))

    def _preprocessing(self, cls, data_path_list, div, cate, dev, chunk_size):
        chunk_offsets = self._split_data(data_path_list, div, chunk_size)
        num_chunks = len(chunk_offsets) # 814 51
        self.logger.info('split data into %d chunks,' % (num_chunks))
        pool = Pool(opt.num_workers)
        try:
            pool.map_async(preprocessing, [(cls,
                                            data_path_list,
                                            div,
                                            self.tmp_chunk_tpl % cidx,
                                            begin,
                                            end,
                                            cate,
                                            dev
                                           )
                                           for cidx, (begin, end) in enumerate(chunk_offsets)]).get(9999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        return num_chunks
    
    def analyze_han(self, api, data):
        product = re_han.sub(' ', data).strip()
        result = []
        if product:
            for w in api.analyze(product):
                condition = True
                bag = []
                for i in w.morphs:
                    if i.tag in ['NNG', 'NNP', 'JKB', 'XSN']:
                        bag.append(i.lex)
                    else:
                        condition *=False
                if condition:
                    result.append(w.lex)
                else:
                    result.extend(bag)

        return result
    
    def analyze_eng(self, data):
        stemmer = SnowballStemmer('english')
        
        product = re_eng.sub(' ', data).strip()
        lower_case = product.lower()
        
        words = lower_case.split()
        words = [w for w in words if not w in stopwords.words('english')]
        words = nltk.pos_tag(words)
        words = [w for w, tag in words if tag in ['NN', 'NNS']]
        words = [stemmer.stem(w) for w in words if len(w)>2]
        
        return words
    
    def analyze(self, data):
        han_result = self.analyze_han(self.api, data)
        eng_result = self.analyze_eng(data)
        result = han_result + eng_result
        return result

    def parse_data(self, label, h, i, cate, dev):
        b, m, s, d = label

        b_index = self.b_vocab.get(b, 0)
        m_index = self.m_vocab.get(m, -1)
        s_index = self.s_vocab.get(s, 0)
        d_index = self.d_vocab.get(d, 0)

        if dev:
            pass
        else:
            if cate == 'a':
                if m_index == -1:
                    return [None] * 2
            elif cate == 's':
                if s_index == 0:
                    return [None] * 2
            elif cate == 'd':
                if d_index == 0:
                    return [None] * 2
            else:
                pass
        
        Y = (b_index, m_index, s_index, d_index)
        
        product = h['product'][i]
        if six.PY3:
            product = product.decode('utf-8')

        words = self.analyze(product)
        words = [w for w in words
                 if len(w) >= opt.min_word_length and len(w) < opt.max_word_length]
        if not words:
            if dev:
                return Y, (np.zeros(opt.max_len, dtype=np.float32), np.zeros(opt.max_len, dtype=np.int32))
            else:
                return [None] * 2

        hash_func = hash if six.PY2 else lambda x: mmh3.hash(x, seed=17)
        x = [hash_func(w) % opt.unigram_hash_size + 1 for w in words]
        xv = Counter(x).most_common(opt.max_len)

        x = np.zeros(opt.max_len, dtype=np.float32)
        v = np.zeros(opt.max_len, dtype=np.int32)
        for i in range(len(xv)):
            x[i] = xv[i][0]
            v[i] = xv[i][1]
        return Y, (x, v)

    def create_dataset(self, g, size):
        shape = (size, opt.max_len)
        g.create_dataset('uni', shape, chunks=True, dtype=np.int32)
        g.create_dataset('w_uni', shape, chunks=True, dtype=np.float32)
        g.create_dataset('pid', (size,), chunks=True, dtype='S12')
        g.create_dataset('img', (size, opt.img_size), chunks=True, dtype=np.float32)
        g.create_dataset('bindex', (size,), chunks=True, dtype=np.int32)
        g.create_dataset('mindex', (size,), chunks=True, dtype=np.int32)
        g.create_dataset('sindex', (size,), chunks=True, dtype=np.int32)
        g.create_dataset('dindex', (size,), chunks=True, dtype=np.int32)

    def init_chunk(self, chunk_size):
        chunk_shape = (chunk_size, opt.max_len)
        chunk = {}
        chunk['uni'] = np.zeros(shape=chunk_shape, dtype=np.int32)
        chunk['w_uni'] = np.zeros(shape=chunk_shape, dtype=np.float32)
        chunk['pid'] = []
        chunk['bindex'] = []
        chunk['mindex'] = []
        chunk['sindex'] = []
        chunk['dindex'] = []
        chunk['num'] = 0
        chunk['img'] = np.zeros(shape=(chunk_size, opt.img_size), dtype=np.float32)
        return chunk

    def copy_chunk(self, dataset, chunk, offset, with_pid_field=False):
        num = chunk['num']
        dataset['uni'][offset:offset + num, :] = chunk['uni'][:num]
        dataset['w_uni'][offset:offset + num, :] = chunk['w_uni'][:num]
        dataset['img'][offset:offset + num, :] = chunk['img'][:num]
        dataset['bindex'][offset:offset + num] = chunk['bindex'][:num]
        dataset['mindex'][offset:offset + num] = chunk['mindex'][:num]
        dataset['sindex'][offset:offset + num] = chunk['sindex'][:num]
        dataset['dindex'][offset:offset + num] = chunk['dindex'][:num]
        
        if with_pid_field:
            dataset['pid'][offset:offset + num] = chunk['pid'][:num]

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    def make_db(self, data_name, output_dir='data/train', cate='a', dev=False, train_ratio=0.8):
        if data_name == 'train':
            div = 'train'
            data_path_list = opt.train_data_list
        elif data_name == 'dev':
            div = 'dev'
            data_path_list = opt.dev_data_list
        elif data_name == 'test':
            div = 'test'
            data_path_list = opt.test_data_list
        else:
            assert False, '%s is not valid data name' % data_name

        all_train = train_ratio >= 1.0
        all_dev = train_ratio == 0.0

        np.random.seed(17)
        self.logger.info('make database from data(%s) with train_ratio(%s)' % (data_name, train_ratio))

        self.load_y_vocab()
        num_input_chunks = self._preprocessing(Data,
                                               data_path_list,
                                               div,
                                               cate,
                                               dev,
                                               chunk_size=opt.chunk_size)
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        if cate == 'a':
            if dev:
                if data_name == 'test':
                    data_fout = h5py.File(os.path.join(output_dir, 'khaiii2_data.h5py'), 'w')
                else:
                    data_fout = h5py.File(os.path.join(output_dir, 'khaiii2_data.h5py'), 'w')
            else:
                data_fout = h5py.File(os.path.join(output_dir, 'khaiii2_data.h5py'), 'w')
            
        else:
            data_fout = h5py.File(os.path.join(output_dir, 'khaiii2_data_'+cate+'.h5py'), 'w')

        
        reader = Reader(data_path_list, div, None, None)
        tmp_size = reader.get_size()
                                               
        train_indices, train_size = self.get_train_indices(tmp_size, train_ratio)
        dev_size = tmp_size - train_size
        if all_dev:
            train_size = 1
            dev_size = tmp_size
        if all_train:
            dev_size = 1
            train_size = tmp_size

        if div == 'train':
            for p in data_path_list:
                print(f"{p} >>>processed") 
        
        train = data_fout.create_group('train')
        dev = data_fout.create_group('dev')
        self.create_dataset(train, train_size)
        self.create_dataset(dev, dev_size)
        self.logger.info('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk_size = opt.db_chunk_size
        chunk = {'train': self.init_chunk(chunk_size),
                 'dev': self.init_chunk(chunk_size)}
        chunk_order = list(range(num_input_chunks))
        np.random.shuffle(chunk_order)
        
        for input_chunk_idx in tqdm.tqdm(chunk_order):
            path = os.path.join(self.tmp_chunk_tpl % input_chunk_idx)
            self.logger.info('processing %s ...' % path)
            data = list(enumerate(cPickle.loads(open(path, 'rb').read())))
            np.random.shuffle(data)
            for data_idx, (pid, y, vw, img) in data:
                b, m, s, d = y
                v, w = vw
                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True
                if v is None:
                    continue
                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['uni'][idx] = v
                c['w_uni'][idx] = w
                c['img'][idx] = img
                c['bindex'].append(b)
                c['mindex'].append(m)
                c['sindex'].append(s)
                c['dindex'].append(d)
                c['num'] += 1
                if not is_train:
                    c['pid'].append(np.string_(pid))
                for t in ['train', 'dev']:
                    if chunk[t]['num'] >= chunk_size:
                        self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                        with_pid_field=t == 'dev')
                        num_samples[t] += chunk[t]['num']
                        chunk[t] = self.init_chunk(chunk_size)
            sample_idx += len(data)
            os.remove(path)
            print(path, ">>>>>>> deleted") # 하나의 파일 처리 후 파일 삭제

        for t in ['train', 'dev']:
            if chunk[t]['num'] > 0:
                self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                with_pid_field=t == 'dev')
                num_samples[t] += chunk[t]['num']

        for div in ['train', 'dev']:
            ds = dataset[div]
            size = num_samples[div]
            shape = (size, opt.max_len)
            ds['uni'].resize(shape)
            ds['w_uni'].resize(shape)
            ds['img'].resize((size, opt.img_size))

        data_fout.close()

        self.logger.info('# of samples on train: %s' % num_samples['train'])
        self.logger.info('# of samples on dev: %s' % num_samples['dev'])
        self.logger.info('data: %s' % os.path.join(output_dir, 'data.h5py'))


if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db})
