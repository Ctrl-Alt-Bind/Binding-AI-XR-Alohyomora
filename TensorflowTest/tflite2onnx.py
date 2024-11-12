import os
import tensorflow as tf
import tf2onnx
from keras._tf_keras.keras.models import load_model

# SavedModel 형식으로 저장할 경로 설정
saved_model_dir = "model.tflite"  # 기존 이름과 다른 이름 사용

# tflite를 ONNX 형식으로 변환
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_tflite(saved_model_dir, output_path=output_path)
print(f"ONNX 모델이 '{output_path}'에 저장되었습니다.")