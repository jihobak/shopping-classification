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
from keras.utils.np_utils import to_categorical
from six.moves import cPickle
from khaiii import KhaiiiApi
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pickle


from misc import get_logger, Option
opt = Option('./config.json')

re_sc = re.compile('[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')
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
            """
            처리 범위 180000 ~ 200000
            end_offset = 200000

            맨끝 처리 파일의 i는 
            i = offset(0) + 199999

            여기에 걸리지 않음
            """
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
        b = h['bcateid'][i] # np.int32 가 자료형이다.
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        #return '%s>%s>%s>%s' % (b, m, s, d)
        return (b, m, s, d)

    def generate(self):
        """
        async로 인해서 블랑킹이 필요없는 경우 
        여러 프로세스로 작업을 넘겨주게 되는데

        이 함수는 주어진 부분을 
        beginoffset 과 endoffset으로 찾아서 해결함.
        크게 두가지 부분을 파악해야 한다.
        1. 해당 begin ~ end 까지가 있는 데이터 파일이 맞는가?
        2. 그 데이터 속에서 해당 begin ~ end에 해당하는 부부인가?
        """
        offset = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')[self.div]
            sz = h['pid'].shape[0]
            if self.begin_offset and offset + sz < self.begin_offset:
                """
                train.chunk.02 을 처리해야하는데
                현재 처리 path가 train.chunk.01 처럼, 이 전인경우
                begin_offset = 200000
                end_offset = 400000
                offset = 0
                sz = 200000

                --> offset = 200000 으로 바뀜
                """
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                """

                1. 처리 범위를 벗어난 데이터일때
                160000 ~ 180000
                begin_offset = 160000
                end_offset = 180000
                sz = 200000
                offset = 200000

                180000 200000 가되서 break



                2. 처리 범위를 벗어난 데이터이지만 전 파일의
                끝 인덱스 처리범위일때
                
                180000 ~ 200000
                train.chunk.01 을 처리해야함,
                train.chunk.01 은 처리가 다된상황
                data_path가 train.chunk.02차례로 넘어옴

                begin_offset = 180000
                end_offset = 200000
                sz = 200000
                offset = 200000
                인 상황이면 다음 로직으로 넘어감
                그러면 아래 로직을 돌게되고 
                offset은 400000이 됨
                그러면 다시 이 로직으로오게되고 break

                """
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    """
                    해당 범위가 들어있는 데이터 파일인것은 맞으나
                    인덱스가 안맞는 부분은 넘김
                    """
                    continue
                bmsd = self.get_class(h, i) # bmsd, type:tuple (b, m, s, d)
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
    """
    y_vocab을 load하고
    Data.preprocessing 메쏘드에 작업을 전달하는 역할이다.
    """
    try:
        cls, data_path_list, div, out_path, begin_offset, end_offset, cate, dev= data
        data = cls()
        data.load_y_vocab()
        data.preprocessing(data_path_list, div, begin_offset, end_offset, out_path, cate, dev)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def build_y_vocab(data):
    try:
        data_path, div = data
        reader = Reader([], div, None, None)
        y_vocab = reader.get_y_vocab(data_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    return y_vocab


class Data:
    y_vocab_path = './data/y_vocab.cPickle' if six.PY2 else './data/y_vocab.py3.cPickle'
    tmp_chunk_tpl = 'tmp2/base.chunk.%s'

    def __init__(self):
        self.logger = get_logger('data')
        self.api = KhaiiiApi()
        with open('./data/train/b_tfidf.pickle', 'rb') as f:
            self.b_tfidf = pickle.load(f)
    
    def get_idf(self, word):
        value = 1
        try:
            index = self.b_tfidf.vocabulary_[str(word)]
            value = self.b_tfidf.idf_[index]
        except KeyError as e:
            pass
        finally:
            return value

    def load_y_vocab(self):
        self.y_vocab = cPickle.loads(open(self.y_vocab_path, 'rb').read())
        self.b_vocab = cPickle.loads(open('./data/b_vocab.cPickle', 'rb').read())
        self.m_vocab = cPickle.loads(open('./data/m_vocab.cPickle', 'rb').read())
        self.s_vocab = cPickle.loads(open('./data/s_vocab.cPickle', 'rb').read())
        self.d_vocab = cPickle.loads(open('./data/d_vocab.cPickle', 'rb').read())
		
    def build_y_vocab(self):
        """
        데이터의 label을 만드는 과정
        ex)
           {'1>1>2>-1': 0, '3>3>4>-1': 1, '5>5>6>-1': 2, '7>7>8>-1': 3 ....}
        """
        pool = Pool(opt.num_workers)
        try:
            rets = pool.map_async(build_y_vocab,
                                  [(data_path, 'train')
                                   for data_path in opt.train_data_list]).get(99999999)
            pool.close()
            pool.join()
            y_vocab = set() #여기는 중복 불가
            # 한 데이터 셋에 대해서 _y_vocab 에는 중복없이 키들이 들어가있음
            for _y_vocab in rets:
                for k in six.iterkeys(_y_vocab):
                    y_vocab.add(k) # 여러 데이터셋에서 키를 추가함으로 다시 set 활용 
            self.y_vocab = {y: idx for idx, y in enumerate(y_vocab)}
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        self.logger.info('size of y vocab: %s' % len(self.y_vocab))
        cPickle.dump(self.y_vocab, open(self.y_vocab_path, 'wb'), 2)

    def _split_data(self, data_path_list, div, chunk_size):
        """
        전체 학습데이터의 개수를 카운트 한뒤
        인덱스 튜플 리스트를 반환한다.
        """
        total = 0
        
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
            
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]
        return chunks

    def preprocessing(self, data_path_list, div, begin_offset, end_offset, out_path, cate, dev):
        """
        tmp/base.chunk.0
        tmp/base.chunk.1 같은 파일을 만든다

        """
        self.div = div
        reader = Reader(data_path_list, div, begin_offset, end_offset)
        rets = []
        for pid, bmsd, h, i, img in reader.generate():
            y, x = self.parse_data(bmsd, h, i, cate, dev)
            if y is None:
                continue
            
            # y = (b, m, s d)
            rets.append((pid, y, x, img)) #
        self.logger.info('sz=%s' % (len(rets)))
        open(out_path, 'wb').write(cPickle.dumps(rets, 2))
        self.logger.info('%s ~ %s done. (size: %s)' % (begin_offset, end_offset, end_offset - begin_offset))

    def _preprocessing(self, cls, data_path_list, div, cate, dev, chunk_size):
        """
        """
        chunk_offsets = self._split_data(data_path_list, div, chunk_size)
        num_chunks = len(chunk_offsets) # 814 51
        self.logger.info('split data into %d chunks, # of classes=%s' % (num_chunks, len(self.y_vocab)))
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
        #print(f"original:{data}")
        product = re_han.sub(' ', data).strip()
        result = []
        if product:
            #print(f"product: {product}")
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
        #print(f"original:{data}")
        stemmer = SnowballStemmer('english')
        
        product = re_eng.sub(' ', data).strip()
        lower_case = product.lower()
        
        words = lower_case.split()
        words = [w for w in words if not w in stopwords.words('english')]
        words = nltk.pos_tag(words)
        words = [w for w, tag in words if tag in ['NN', 'NNS']]
        words = [stemmer.stem(w) for w in words if len(w)>2]
        
        return words
    
    def analyze2(self, data):
        han_result = self.analyze_han(self.api, data)
        eng_result = self.analyze_eng(data)
        result = han_result + eng_result
        return result

    def analyze(self, data):
        product = re_sc.sub(' ', data).strip()
        result = []
        if product:
            #api = KhaiiiApi()
            for w in self.api.analyze(product):
                for i in w.morphs:
                    if i.tag in ['NNG', 'NNP']:
                        result.append(i.lex)
        
        return result

    def parse_data(self, label, h, i, cate, dev):
        #Y = self.y_vocab.get(label)
        b, m, s, d = label # 실제 데이터에는 b, m, s, d 모두 -1이 있을 수 있다. ex) dev, test

        b_index = self.b_vocab.get(b, 0)
        m_index = self.m_vocab.get(m, -1)
        s_index = self.s_vocab.get(s, 0)
        d_index = self.d_vocab.get(d, 0)

        # dev 나 tes 데이터에 대해서 샘플들을 필터링 할 필요가 없기 때문에
        # 아래 조건문들은 필요가 없다.
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
        #product = re_sc.sub(' ', product).strip().split()
        #words = [w.strip() for w in product]
        words = self.analyze2(product)
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
            v[i] = xv[i][1] #self.b_tfidf.idf_[self.b_tfidf.vocabulary_[str(xv[i][0])]] #self.get_idf(xv[i][0]) #xv[i][1]
        return Y, (x, v)

    def create_dataset(self, g, size, num_classes):
        shape = (size, opt.max_len)
        g.create_dataset('uni', shape, chunks=True, dtype=np.int32)
        g.create_dataset('w_uni', shape, chunks=True, dtype=np.float32)
        #g.create_dataset('cate', (size, num_classes), chunks=True, dtype=np.int32)
        g.create_dataset('pid', (size,), chunks=True, dtype='S12')
        g.create_dataset('img', (size, opt.img_size), chunks=True, dtype=np.float32)
        g.create_dataset('bindex', (size,), chunks=True, dtype=np.int32)
        g.create_dataset('mindex', (size,), chunks=True, dtype=np.int32)
        g.create_dataset('sindex', (size,), chunks=True, dtype=np.int32)
        g.create_dataset('dindex', (size,), chunks=True, dtype=np.int32)

    def init_chunk(self, chunk_size, num_classes):
        chunk_shape = (chunk_size, opt.max_len)
        chunk = {}
        chunk['uni'] = np.zeros(shape=chunk_shape, dtype=np.int32)
        chunk['w_uni'] = np.zeros(shape=chunk_shape, dtype=np.float32)
        #chunk['cate'] = np.zeros(shape=(chunk_size, num_classes), dtype=np.int32)
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
        #dataset['cate'][offset:offset + num] = chunk['cate'][:num]
        dataset['bindex'][offset:offset + num] = chunk['bindex'][:num]
        dataset['mindex'][offset:offset + num] = chunk['mindex'][:num]
        dataset['sindex'][offset:offset + num] = chunk['sindex'][:num]
        dataset['dindex'][offset:offset + num] = chunk['dindex'][:num]
        
        if with_pid_field:
            dataset['pid'][offset:offset + num] = chunk['pid'][:num]

    def copy_bulk(self, A, B, offset, y_offset, with_pid_field=False):
        num = B['cate'].shape[0]
        y_num = B['cate'].shape[1]
        A['uni'][offset:offset + num, :] = B['uni'][:num]
        A['w_uni'][offset:offset + num, :] = B['w_uni'][:num]
        A['img'][offset:offset + num, :] = B['img'][:num]
        #A['cate'][offset:offset + num, y_offset:y_offset + y_num] = B['cate'][:num]
        A['bindex'][offset:offset + num] = B['bindex'][:num]
        A['mindex'][offset:offset + num] = B['mindex'][:num]
        A['sindex'][offset:offset + num] = B['sindex'][:num]
        A['dindex'][offset:offset + num] = B['dindex'][:num]
        
        if with_pid_field:
            A['pid'][offset:offset + num] = B['pid'][:num]

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    def make_db(self, data_name, output_dir='data/train', cate='a', dev=False, train_ratio=0.8):
        """
        python data.py make_db train data/train a False 0.8
        python data.py make_db train data/train s False 0.8
        python data.py make_db train data/train d False 0.8
        
        python data.py make_db dev ./data/dev a True --train_ratio=0.0
        
        python data.py make_db test ./data/test a True --train_ratio=0.0
        """
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
        
        #num_input_chunks = 814 # dev 인경우 51
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        if cate == 'a':
            if dev:
                data_fout = h5py.File(os.path.join(output_dir, 'dev_khaiii2_data.h5py'), 'w')
            else:
                data_fout = h5py.File(os.path.join(output_dir, 'khaiii2_data_12000.h5py'), 'w')
            
        else:
            if dev:
                if data_name == 'test':
                    data_fout = h5py.File(os.path.join(output_dir, 'test_khaiii2_data_120000_bm'+cate+'.h5py'), 'w')
                else:    
                    data_fout = h5py.File(os.path.join(output_dir, 'dev_khaiii2_data_120000_bm'+cate+'.h5py'), 'w') #없애도될듯?
                
            else:
                data_fout = h5py.File(os.path.join(output_dir, 'khaiii2_data_120000_bm'+cate+'.h5py'), 'w')
        
        meta_fout = open(os.path.join(output_dir, 'meta'), 'wb')

        
        reader = Reader(data_path_list, div, None, None) ####
        tmp_size = reader.get_size() ####
        
        ### parse_data 와 아래 tmp_size를 변경해서
        ### bms, bmdsd 데이터를 만들것. 
        #tmp_size = 8134818
        #num_dsize = 605995 
        #num_dindex_dev = 151862
        #num_ssize = 5014987
        #num_sindex_dev = 1253557
                                               
        train_indices, train_size = self.get_train_indices(tmp_size, train_ratio)
        dev_size = tmp_size - train_size
        if all_dev:
            train_size = 1
            dev_size = tmp_size # 507783
        if all_train:
            dev_size = 1
            train_size = tmp_size

        if div == 'train':
            for p in data_path_list:
                # os.remove(p) # training data 삭제
                print(f"{p} >>>processed") 
        
        train = data_fout.create_group('train')
        dev = data_fout.create_group('dev')
        self.create_dataset(train, train_size, len(self.y_vocab))
        self.create_dataset(dev, dev_size, len(self.y_vocab))
        self.logger.info('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk_size = opt.db_chunk_size
        chunk = {'train': self.init_chunk(chunk_size, len(self.y_vocab)),
                 'dev': self.init_chunk(chunk_size, len(self.y_vocab))}
        chunk_order = list(range(num_input_chunks))
        np.random.shuffle(chunk_order)
        print("fuck")
        for input_chunk_idx in tqdm.tqdm(chunk_order):
            """
            여기서 부터 임시 청크 파일들을 하나 하나 씩 열어서
            최종적 결과물인 data.hd5py 로 하나로 만들어 내는 것.
            """
            path = os.path.join(self.tmp_chunk_tpl % input_chunk_idx)
            self.logger.info('processing %s ...' % path)
            data = list(enumerate(cPickle.loads(open(path, 'rb').read())))
            np.random.shuffle(data)
            for data_idx, (pid, y, vw, img) in data:
                #if y is None:
                #    continue
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
                c['uni'][idx] = v # 어떤 단어들이있는가 [10176, 30012, ....]
                c['w_uni'][idx] = w # 위의 각 단어들의 빈도수 [1,1,2, ....]
                #c['cate'][idx] = y # 카테고리, 정답 [0, 1, 0, .... 0]
                c['img'][idx] = img # img 추가
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
                        chunk[t] = self.init_chunk(chunk_size, len(self.y_vocab))
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
            #ds['cate'].resize((size, len(self.y_vocab)))

        data_fout.close()
        meta = {'y_vocab': self.y_vocab}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.logger.info('# of samples on train: %s' % num_samples['train'])
        self.logger.info('# of samples on dev: %s' % num_samples['dev'])
        self.logger.info('data: %s' % os.path.join(output_dir, 'data.h5py'))
        self.logger.info('meta: %s' % os.path.join(output_dir, 'meta'))


if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db,
               'build_y_vocab': data.build_y_vocab})
