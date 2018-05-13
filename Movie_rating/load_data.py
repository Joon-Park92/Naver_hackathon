#-*- coding: utf-8 -*-

import zipfile
import json
import pandas as pd
import tqdm


## 원본파일은 Json 파일

# From jsonfiles
def load_test_df():
    with zipfile.ZipFile('/media/disk1/public_milab/Udacity/data/Movie_rate/Naver_Movie_Rate.zip') as f:
        namelist = f.namelist()
        df = pd.read_json(f.read(namelist[0]))
    return df

# From jsonfiles
def load_data_df():
    with zipfile.ZipFile('/media/disk1/public_milab/Udacity/data/Movie_rate/Naver_Movie_Rate.zip') as f:
        namelist = f.namelist()
        df = pd.DataFrame()
        for name in tqdm.tqdm(namelist):
            df = df.append(pd.read_json(f.read(name)))
    
    return df

## csv로 저장한 파일 불러오기 
def load_original_data():
    df = pd.read_csv('/media/disk1/public_milab/Udacity/data/Movie_rate/Naver_Movie_Rate.csv', encoding='utf8')
    return df

## Konlpy 로 전처리한 Review 불러오기
def load_konlpy_phrase_review_inner():
    with open('/media/disk1/public_milab/Udacity/Naver_hackathon/rating/train_konlpy_phrase.txt', 'r') as f:
        return f.read().split('\n')
    
def load_konlpy_phrase_review():
    data = load_original_data().xs(['review', 'rating'], axis = 1)
    data.dropna(inplace =True)
    konlpy_data = load_konlpy_phrase_review_inner()
    df = pd.DataFrame({'review': konlpy_data[:-1] , 'rating' : data.rating} )
    return df