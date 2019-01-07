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
from collections import defaultdict
import fire
import h5py
import numpy as np
import six
from six.moves import zip, cPickle
from tqdm import tqdm

def get_size(data_path, div):
    h = h5py.File(data_path)[div]
    size = h['img'].shape[0]
    return size


def toss_answer(data_path, div):
    h = h5py.File(data_path)[div]
    size = h['cate'].shape[0]
    for i in range(size):
        yield np.argmax(h['cate'][i])


def toss_chunk_answer(data_path, div):
    h = h5py.File(data_path)[div]
    size = h['img'].shape[0]
    chunk_sz = 1000000

    chunk_ix = [(i, min(i+chunk_sz, size)) for i in range(0, size, chunk_sz)]

    for start, end in chunk_ix:
        b = h['bindex'][start:end]
        m = h['mindex'][start:end]
        s = h['sindex'][start:end]
        d = h['dindex'][start:end]

        answer = [(a, b, c, d) for a, b, c, d in zip(b, m, s, d)]
        yield answer
        #yield np.argmax(h['cate'][start:end], axis=1)


def evaluate(predict_path, data_path, div, y_vocab_path, log_path):
    """
    python evaluate.py evaluate onlym_khaiii_textimg1024_predict.tsv ./data/train/data.h5py dev ./data/y_vocab.py3.cPickle onlym_khaiii_textimg1024_score.txt
    
    #khaiii textimg 1024
    python evaluate.py evaluate valid_khaiii_textimg1024_predict.tsv ./data/train/khaiii_data.h5py dev ./data/y_vocab.py3.cPickle valid_khaiii_textimg1024_score.txt
    
    #khaii2 textimg 1024
    python evaluate.py evaluate valid_khaiii2_textimg1024_predict.tsv ./data/train/khaiii2_data.h5py dev ./data/y_vocab.py3.cPickle valid_khaiii2_textimg1024_score.txt
    
    #khaiii2_12_512textimgdrop_relu_cw_1024
    python evaluate.py evaluate valid_khaiii2_12_512textimgdrop_relu_cw_1024_predict.tsv ./data/train/khaiii2_data_120000.h5py dev ./data/y_vocab.py3.cPickle valid_khaiii2_12_512textimgdrop_relu_cw_1024_score.txt
    """
    #h = h5py.File(data_path, 'r')[div]
    y_vocab = cPickle.loads(open(y_vocab_path, 'rb').read())
    inv_y_vocab = {v: k for k, v in six.iteritems(y_vocab)}

    b_vocab = cPickle.loads(open("./data/b_vocab.cPickle", 'rb').read())
    m_vocab = cPickle.loads(open("./data/m_vocab.cPickle", 'rb').read())
    s_vocab = cPickle.loads(open("./data/s_vocab.cPickle", 'rb').read())
    d_vocab = cPickle.loads(open("./data/d_vocab.cPickle", 'rb').read())
    
    inv_b_vocab = {i: s for s, i in six.iteritems(b_vocab)}
    inv_m_vocab = {i: s for s, i in six.iteritems(m_vocab)}
    inv_s_vocab = {i: s for s, i in six.iteritems(s_vocab)}
    inv_d_vocab = {i: s for s, i in six.iteritems(d_vocab)}

    fin = open(predict_path, 'r')
    hit, n = defaultdict(lambda: 0), defaultdict(lambda: 0)
    print('loading ground-truth...')
    #CATE = np.argmax(h['cate'], axis=1)
    
    size = get_size(data_path, div)
    #CATE = toss_answer(data_path, div)
    
    bomb = toss_chunk_answer(data_path, div)
    for bx in bomb:
        for p, y in tqdm(zip(fin, bx), desc='bomb', total=len(list(bx))):
            # format y = (b, m, s, d) this is answer
            pid, b, m, s, d = p.split('\t')
            b, m, s, d = list(map(int, [b, m, s, d]))      # 나의 prediction
            #gt = list(map(int, inv_y_vocab[y].split('>'))) # 정답
            
            gt_b = inv_b_vocab[y[0]]
            gt_m = inv_m_vocab[y[1]]
            gt_s = inv_s_vocab[y[2]]
            gt_d = inv_d_vocab[y[3]]

            gt = [gt_b, gt_m, gt_s, gt_d]

            for depth, _p, _g in zip(['b', 'm', 's', 'd'],
                                    [b, m, s, d],
                                    gt):
                if _g == -1:
                    continue
                n[depth] = n.get(depth, 0) + 1 # 총 개수 파악
                if _p == _g:
                    hit[depth] = hit.get(depth, 0) + 1 # 맞은 개수 기록
    
    with open(log_path, 'w') as f:
        for d in ['b', 'm', 's', 'd']:
            if n[d] > 0:
                print('%s-Accuracy: %.3f(%s/%s)' % (d, hit[d] / float(n[d]), hit[d], n[d]))
                f.write('%s-Accuracy: %.3f(%s/%s) \n' % (d, hit[d] / float(n[d]), hit[d], n[d]))
        score = sum([hit[d] / float(n[d]) * w
                    for d, w in zip(['b', 'm', 's', 'd'],
                                    [1.0, 1.2, 1.3, 1.4])]) / 4.0
        print('score: %.3f' % score)
        f.write('score: %.3f\n' % score)
        


if __name__ == '__main__':
    fire.Fire({'evaluate': evaluate})
