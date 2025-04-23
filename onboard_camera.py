import cv2

# replace ip/user/pass as needed
rtsp_url = "rtsp://admin:admin@192.168.1.10/color"
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Color Stream", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
