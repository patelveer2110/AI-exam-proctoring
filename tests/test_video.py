import cv2
import requests

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    _, img_encoded = cv2.imencode('.jpg', frame)

    response = requests.post(
        "http://127.0.0.1:8000/video/analyze",
        files={"file": img_encoded.tobytes()}
    )

    print(response.json())

    cv2.imshow("Test", frame)
    if cv2.waitKey(1) == 27:
        break