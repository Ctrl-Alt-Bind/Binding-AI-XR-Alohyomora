import numpy as np
from pathlib import Path
import tensorflow as tf
import keras

# 데이터 불러오기 (처음에 인터넷에서 데이터셋 다운로드 후, 캐시 디렉터리 /.keras/datasets에 저장함.)
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()


# 절대 경로 설정
import os
train_path = os.path.abspath("C:/Users/aloho/문서/Github/Binding-AI-XR-Alohyomora/TensorflowTest/cifar10DataSet/cifar10/train")
test_path = os.path.abspath("C:/Users/aloho/문서/Github/Binding-AI-XR-Alohyomora/TensorflowTest/cifar10DataSet/cifar10/test")

# 최상위 디렉터리 구조 확인
print("train 경로 내용:", os.listdir(train_path))

# 각 클래스 디렉터리 내용 확인
for subdir in os.listdir(train_path):
    sub_path = os.path.join(train_path, subdir)
    if os.path.isdir(sub_path):
        print(f"{subdir} 클래스 디렉터리 내용:")#, os.listdir(sub_path))
    else:
        print(f"{subdir}는 폴더가 아닙니다.")
        
try:
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        train_path,
        seed=123,
        image_size=(32, 32),
        batch_size=32
    )
except Exception as e:
    print("오류 발생:", e)
    
# Keras 함수 호출 시 경로를 문자열로 변환하여 전달
train_dataset = keras.preprocessing.image_dataset_from_directory(
    train_path,     # 문자열로 변환하여 전달
    seed=123,
    image_size=(32, 32),
    batch_size=32
)

test_dataset = keras.preprocessing.image_dataset_from_directory(
    test_path,      # 문자열로 변환하여 전달
    seed=123,
    image_size=(32, 32),
    batch_size=32
)
