import cv2
import ultralytics
from ultralytics import YOLO
import torch
import os

#ultralytics.checks()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# base model load
model = YOLO('yolov8n.pt')

results_train = model.train(data='C:/Users/aloho/Github/Binding-AI-XR-Alohyomora/YOLOv8/Yolo Test/custom.yaml', epochs = 5)