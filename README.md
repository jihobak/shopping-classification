# [Online shopping mall product category classification competition sponsored by KaKao](https://arena.kakao.com/c/1)
---


Participants deal with almost 100GB data and should classify a huge amounts of products using their own algorithms. Data contains text data and Images. I apply deep learning technique and placed 14nd as a result.[[leader board](https://arena.kakao.com/c/1/leaderboard)]


---

### 추가 라이브러리 사용여부
- 최근 kakao에서 공개한 [khaiii 형태소 분석기](https://github.com/kakao/khaiii)를 사용하였습니다.
- [nltk](https://www.nltk.org/)를 사용하였습니다.

### 모델 설명 및 요약
- 4개의 카테고리 b, m, s, d 별로 각각 예측하는 분류기를 만듭니다.
- `product, img_feat`을 feature로 활용합니다.
-  상품명에 대해서 khaiii형태소 분석기와 nltk를 활용해서 전처리합니다. 

### 실행 방법

0. 데이터의 위치
    - 내려받은 데이터의 위치는 소스가 실행될 디렉토리의 상위 디렉토리로(../) 가정되어 있습니다.

1. 카테고리별 label데이터를 담고있는 사전만들기
    - `python make_vocab.py` 을 실행하면 아래 4개의 파일이 생성됩니다.
     - b_vocab.cPickle
     - m_vocab.cPickle
     - s_vocab.cPickle
     - d_vocab.cPickle

2. 학습데이터 생성
    - b, m 카테고리를 위한 트레이닝 데이터 만들기 -> `python data.py make_db train data/train a False 0.8`
     - `./data/train/khaiii2_data.h5py` 가 생성이 됩니다.
    - s 카테고리를 위한 트레이닝 데이터 만들기 -> `python data.py make_db train data/train s False 0.8`
     - `./data/train/khaiii2_data_s.h5py`가 생성이 됩니다.
    - d 카테고리를 위한 트레이닝 데이터 만들기 -> `python data.py make_db train data/train d False 0.8`
     - `./data/train/khaiii2_data_d.h5py`가 생성이 됩니다.

3. 최종 prediction 을 위한 테스트 데이터 만들기
    - `python data.py make_db test ./data/test a True --train_ratio=0.0`
     - `./data/test/khaiii2_data.h5py`가 생성이 됩니다.[여기](https://drive.google.com/open?id=1a0M5yrLn64R5HqVOe6j71IMiydQZX_9C)(50G)에서 다운가능합니다.

4. 카테고리별로 모델 훈련하기
    - `python train.py train_textimg ./data/train khaiii2_data.h5py ./model/train b 1024`
    - `python train.py train_textimg ./data/train khaiii2_data.h5py ./model/train m 1024`
    - `python train.py train_textimg ./data/train khaiii2_data_s.h5py ./model/train s 1024`
    - `python train.py train_textimg ./data/train khaiii2_data_d.h5py ./model/train d 1024`
    - 학습된 모델 파일은 [여기](https://drive.google.com/open?id=1aZmHS96rjOWBtVxSO1p7YZfU6XV1FMCT)에서 다운로드 하실 수 있습니다. 총 4개의 파일을`./data/train`에 위치해주시면 됩니다.

5. Prediction 결과 파일 생성 
    - 4번과정 후 './model/train'에 위치해있는 모델파일들 중에서 각 카테고리별 best 모델들의 이름을 `inference.py`에
      수정해야합니다.(혹시나 다를 가능성이 있기때문에, 모델을 다운로드 하는경우는 상관없습니다.)
    - `python inference.py predict_all ./data/train ./model/train ./data/test/ dev test_predict.tsv`
     - `test_predict.tsv`파일이 생성됩니다.
