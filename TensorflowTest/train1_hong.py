import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.datasets import cifar10

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 정규화
x_train = x_train / 255.0
x_test = x_test /255.0

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Image Data Generator : keras에서 제공하는 데이터 증강 도구로
# 이미지에 여러 변형을 적용해 새로운 학습 데이터 생성
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# x_train 데이터의 첫 번째 차원(항목)을 의미하며, 이는 x_train에 포함된 샘플의 총 개수이다.
augment_size = int(x_train.shape[0])

# 데이터 복사본 만들기
x_augmented = x_train.copy()
y_augmented = y_train.copy()

# 설정된 변환을 기반으로 새로운 이미지 배치 생성
#x_augmented, y_augmented = gen.flow(x_augmented)