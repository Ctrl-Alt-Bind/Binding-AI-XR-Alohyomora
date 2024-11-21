from ultralytics import YOLO
import cv2

# YOLOv8n 모델 불러오기 : 사전 학습된 모델
model = YOLO('yolov8l.pt')

# 이미지 경로 설정
image_path = 'C:\\Users\\aloho\\Github\\Binding-AI-XR-Alohyomora\\YOLOv8\\Yolo Test\\dog.jpg'

# 이미지 로드
img = cv2.imread(image_path)

# 객체 탐지 수행
results = model(image_path)

# 감지된 객체를 이미지에 표시
annotated_img = results[0].plot()

for result in results:
    boxes = result.boxes  # 바운딩 박스 정보
    for box in boxes:
        cls = box.cls  # 클래스 인덱스
        conf = box.conf  # 신뢰도
        xyxy = box.xyxy  # 바운딩 박스 좌표
        print(f'클래스: {cls}, 신뢰도: {conf}, 좌표: {xyxy}')
        
        
# 결과 이미지 저장
output_path = 'annotated_image.jpg'
cv2.imwrite(output_path, annotated_img)
        
# 결과 이미지 표시
cv2.imshow('Detected Objects' , annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()