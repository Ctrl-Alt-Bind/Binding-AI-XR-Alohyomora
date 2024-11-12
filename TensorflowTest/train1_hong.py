import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.datasets import cifar10

# 데이터셋 불러오기 x(pixel data), y(label)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 정규화 (각 픽셀 값을 255로 나누어 0에서 1 사이로 변환, 전처리 작업)
x_train = x_train / 255.0
x_test = x_test /255.0

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# ============ 데이터 증강 및 배치 생성 ============

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# region Image Data Generator : keras에서 제공하는 데이터 증강 도구
# 이미지에 여러 변형을 적용해 새로운 학습 데이터 생성
# 회전, 가로/세로 이동, 좌우 반전 -> 과적합 줄이기
# endregion
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
x_augmented, y_augmented = gen.flow(x_augmented, y_augmented,
                                    batch_size=augment_size,
                                    shuffle=False).__next__()

# 원본 데이터와 증강된 데이터를 결합하여 학습 데이터의 양을 늘린다.
x_train = np.concatenate((x_train, x_augmented))
y_train= np.concatenate((y_train, y_augmented))

# 보강된 학습 데이터, 정답 데이터를 랜덤하게 섞음
s = np.arange(x_train.shape[0])
np.random.shuffle(s)

x_train = x_train[s]
y_train = y_train[s]

# region 신경망 모델 정의
# Sequential : Sequential 모델로 레이어를 순차적으로 쌓는 단순한 구조의 모델이다.
# Conv2D : 합성곱 층으로, 이미지에서 특징을 추출한다. 필터(커널)의 수와 크기를 통해 이미지에서 다양한 특징을 학습한다.
# MaxPooling2D : 풀링 층으로, 공간적 크기를 줄이면서도 중요한 특징만을 추출해 파라미터 수와 연산량을 줄이는 역할을 한다.
# Dropout : 과적합을 방지하기 위해 일부 노드를 임의로 비활성화한다.
# Flatten : CNN의 마지막에서 3차원 텐서를 1차원 벡터로 변환하여 Dense 레이어에 연결할 수 있게 한다.
# Dense : 완전 연결층으로, 신경망의 일반적인 레이어이다.
# Activation : 활성화 함수로 ReLU, SoftMax가 있다.
# endregion
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras._tf_keras.keras import Input

# 레이어를 순차적으로 쌓는 Sequential 모델을 사용
model = Sequential()

# 첫 번째 레이어로 입력 형태 지정
model.add(Input(shape=(32,32,3))) 

# region 합성곱 층(Convolutional Layer) 1
# Conv2D(32, (3,3), activation='relu', padding='same') -> 
#   32 : 32개의 필터를 사용해서 입력 이미지의 각 부분에서 특징을 추출한다.
#   (3,3) : 커널크기 (3,3)으로 각 필터가 이미지에서 3x3 크기로 이동하며 특징을 추출한다. 
#   activation= 'relu : ReLU로 비선형성 추가. 
#   padding='same' : 패딩은 same으로 지정해 입출력 크기를 동일하게 유지
# Batch Normalization 추가
# MaxPooling2D((2,2)) -> 풀링 층을 추가하여 입력 이미지를 (2,2) 크기로 다운 샘플링: 공간적 크기를 줄이면서 중요한 정보 남기기
# Dropout(0.25) -> 랜덤하게 노드 비활성화 : 과적합 방지
# endregion
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))     
model.add(BatchNormalization())  
model.add(MaxPooling2D((2,2)))                                      
model.add(Dropout(0.3))                                            

# 합성곱 층 2
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))     # 필터 개수를 64개로 늘려 더 복잡한 특징을 학습하도록 함.
model.add(BatchNormalization())  
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

# 합성곱 층 3
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))    # 필터 개수를 128개로 늘려 더 복잡한 특징을 학습하도록 함.
model.add(BatchNormalization())  
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

# 특성맵을 벡터로 변환
model.add(Flatten())                                                # CNN의 출력인 3차원 텐서를 1차원 벡터로 변환 완전 연결 층(Dense Layer)에 연결하기 위해 필요

# 완전 연결 층 (은닉층)
model.add(Dense(256, activation='relu'))                            # 128개의 노드를 가진 은닉층
model.add(BatchNormalization())  
model.add(Dropout(0.5))

# 출력층
model.add(Dense(10, activation='softmax'))                          # 최종 출력층으로, 10개의 클래스에 대한 확률을 출력, softmax 활성화 함수를 통해 각 클래스에 대한 확률 값을 구하여 이미지가 어느 클래스에 속하는지 분류


# region 모델 컴파일
# 손실함수 loss=sparse_categorical_crossentropy : 다중 클래스 분류 문제에서 사용
# 최적화 알고리즘 Adam : 경사 하강법을 개선한 최적화 알고리즘
# 평가지표 metrics : accuracy 정확도를 측정
# endregion
from keras._tf_keras.keras.optimizers import Adam
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

# region 모델 학습(model.fit)
# batch_size=256: 한 번에 256개의 데이터 샘플을 사용하여 가중치를 업데이트
# epochs=20: 전체 데이터셋을 20번 반복 학습
# validation_data=(x_test, y_test): 학습 도중에 모델 성능을 검증할 데이터를 제공, 학습 단계마다 모델이 이 데이터를 사용하여 현재 정확도를 확인
# validation_split=0.1: 훈련 데이터의 10%를 검증용으로 사용
# history 객체 : 학습 중 기록된 손실 및 정확도 값을 저장한다. 이후 시각화에 사용됨.
# endregion
history = model.fit(x_train,y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test),validation_split=0.1)

# 모델 평가 (model.evaluate) : 평가 단계에서는 학습되지 않은 테스트 데이터(x_test, y_test)를 사용하여 모델의 최종 성능을 확인
model.evaluate(x_test, y_test)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Trend')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='best')
plt.grid()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Trend')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='best')
plt.grid()
plt.show()

# 모델을 .tflite 형식으로 저장
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)