from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import firebase_admin
from firebase_admin import credentials, db
import threading

# Firebase 초기화
cred = credentials.Certificate("your database.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'your databaseURL'
})

# Picamera2 초기화
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

# 사람 감지 및 결과 저장 함수
def detect_and_store_person_count(image):
    model = YOLO('yolov8n.pt')  # 'yolov8s.pn' 모델 사용
    results = model(image)
    person_count = sum(1 for r in results[0].boxes if r.cls == 0)  # 클래스 0은 사람을 나타낸다. 그래서 사람의 수만 측정
    ref = db.reference('cam2')
    ref.set(person_count)
    annotated_image = results[0].plot()
    cv2.imwrite('annotated_image.jpg', annotated_image)
    print(f"Detected {person_count} persons and saved the annotated image.")

# 이미지 캡처 및 처리 함수
def capture_and_process_image():
    image = picam2.capture_array()
    cv2.imwrite('captured_image.jpg', image)
    detect_and_store_person_count(image)

# Firebase 데이터 변화 감지 리스너 콜백 함수
def on_sensor_data_change(event):
    if event.data:
        capture_and_process_image()
        sensor_data_ref.set({})  # sensor_data 폴더 비우기

# Firebase 리스너 설정
sensor_data_ref = db.reference('sensor_data')
sensor_data_ref.listen(on_sensor_data_change)

# 메인 루프 (리스너가 백그라운드에서 실행되도록 유지)
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Script stopped by user.")
finally:
    picam2.stop()
