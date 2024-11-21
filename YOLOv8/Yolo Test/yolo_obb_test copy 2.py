if __name__ == '__main__':
    # 멀티프로세싱 모듈 초기화 (Windows 환경에서 필수)
    import multiprocessing
    multiprocessing.freeze_support()

    # 필요한 라이브러리 임포트
    import os
    import torch
    from ultralytics import YOLO

    # GPU 환경 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU 장치 순서 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 사용할 GPU 설정 (0번 GPU 사용)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능 여부 확인
    print('Device:', device)
    
    # GPU 사용 가능 여부 디버깅
    if device.type == 'cuda':
        print('Current CUDA device:', torch.cuda.current_device())
        print('Count of GPUs:', torch.cuda.device_count())
    else:
        print("CUDA GPU를 사용할 수 없습니다. CPU로 진행합니다.")

    # YOLO 모델 불러오기
    model = YOLO('yolov8m.pt')  # 사전 학습된 YOLOv8 모델 로드

    # 학습 시작
    try:
        results = model.train(
            data='C:/Users/aloho/Github/Binding-AI-XR-Alohyomora/YOLOv8/Yolo Test/custom.yaml',  # 데이터셋 yaml 경로
            epochs=10,  # 학습 에폭
            imgsz=640,  # 입력 이미지 크기
            device=device  # 디바이스 설정
        )
    except FileNotFoundError as e:
        print(f"파일 경로 오류: {e}")
    except RuntimeError as e:
        print(f"런타임 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
