import cv2
from PIL import Image
import random

# replace ip/user/pass as needed
rtsp_url = "rtsp://admin:admin@192.168.1.10/color"
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    img = Image.fromarray(frame, 'RGB')
    img_name = 'onboard_camera'+str(random.random()*10000)+'.png'
    img.save('real/data/onboard/' +img_name)
    if not ret:
        break
    cv2.imshow("Color Stream", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
