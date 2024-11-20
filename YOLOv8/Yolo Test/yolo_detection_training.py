from ultralytics import YOLO
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np
from tqdm import tqdm

# with open('./ultralytics/datasets/coco.yaml') as f:
#     coco = yaml.load(f, Loader=yaml.FullLoader)
#     class_names = coco['names']
# COCO 클래스 직접 정의 (임시)
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# YOLOv8n 모델 불러오기 : 사전 학습된 모델
model = YOLO('yolov8m.pt')

# # Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')

def draw_bbox(draw, bbox, label, color=(0, 255, 0, 255), confs=None, size=15):
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    draw.rectangle(bbox, outline=color, width =3)
    def set_alpha(color, value):
        background = list(color)
        background[3] = value
        return tuple(background)
    background = set_alpha(color, 50)
    draw.rectangle(bbox, outline=color, fill=background, width =3)
    background = set_alpha(color, 150)
    text = f"{label}" + ("" if confs==None else f":{conf:0.4}")
    text_bbox = bbox[0], bbox[1], bbox[0]+len(text)*10, bbox[1]+25
    draw.rectangle(text_bbox, outline=color, fill=background, width =3)
    draw.text((bbox[0]+5, bbox[1]+5), text, (0,0,0), font=font)

color = []
n_classes = 80
for _ in range(n_classes):
    c = list(np.random.choice(range(256), size=3)) + [255]
    c = tuple(c)
    color.append(c)
    
img = Image.open("./bus.jpg")
img = img.resize((640, 640))
width, height = img.size
draw = ImageDraw.Draw(img, 'RGBA')

for result in results:
    result = result.cpu()
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    # result.boxes.xywh   # box with xywh format, (N, 4)
    xyxys = result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    confs = result.boxes.conf   # confidence score, (N, 1)
    clss = result.boxes.cls    # cls, (N, 1)

    xyxys = xyxys.numpy()
    clss = map(int, clss.numpy())
    for xyxy, conf, cls in zip(xyxys, confs, clss):
        xyxy = [xyxy[0]*width, xyxy[1]*height, xyxy[2]*width, xyxy[3]*height]
        draw_bbox(draw, bbox=xyxy, label=class_names[cls], color=color[cls], confs=confs, size=15)
    img.show()

np.random.seed(724)

# Data set 경로
dir_main = "C:/Users/aloho/Github/Binding-AI-XR-Alohyomora/DataSet/indoor_object_detection_dataset"
filenames_image = glob(f"{dir_main}/train/images/*.jpg")
filenames_label = [filename.replace('images', 'labels').replace('jpg', 'txt') for filename in filenames_image]

cnt = 0
classes = ["door", "openedDoor", "cabinetDoor", "refrigeratorDoor", "window", "chair", "table", "cabinet", "sofa/couch", "pole"]

color = []
for _ in range(10):
    c = list(np.random.choice(range(256), size=3)) + [255]
    c = tuple(c)
    color.append(c)

cnt = 5
for filename_image, filename_label in tqdm(zip(filenames_image, filenames_label)):
    img = Image.open(filename_image)
    img = img.resize((640, 640))
    width, height = img.size
    draw = ImageDraw.Draw(img, 'RGBA')
    with open(filename_label, 'r') as f:
        labels = f.readlines()
        labels = list(map(lambda s: s.strip().split(), labels))
    for label in labels:
        cls = int(label[0])+1
        x, y, w, h = map(float, label[1:])
        x1, x2 = width * (x-w/2), width * (x+w/2)
        y1, y2 = height * (y-h/2), height * (y+h/2)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        draw_bbox(draw, bbox=(x1, y1, x2, y2), label=classes[cls], color=color[cls], size=15)
    img.show()

    cnt -= 1
    if cnt ==0:
        break

results_train = model.train(data='C:/Users/aloho/Github/Binding-AI-XR-Alohyomora/YOLOv8/Yolo Test/custom.yaml', epochs = 10)
    
results_train = model.val()