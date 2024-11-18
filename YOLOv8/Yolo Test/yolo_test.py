from ultralytics import YOLO

# YOLOv8n 모델 불러오기 : 사전 학습된 모델
model = YOLO('yolov8n.pt')

# 이미지 경로 설정
image_path = 'C:\\Users\\aloho\\Github\\Binding-AI-XR-Alohyomora\\YOLOv8\\Yolo Test\\fruit.jpg'

# 객체 탐지 수행
results = model(image_path)

# 결과 시각화

for result in results:
    boxes = result.boxes  # 바운딩 박스 정보
    for box in boxes:
        cls = box.cls  # 클래스 인덱스
        conf = box.conf  # 신뢰도
        xyxy = box.xyxy  # 바운딩 박스 좌표
        print(f'클래스: {cls}, 신뢰도: {conf}, 좌표: {xyxy}')