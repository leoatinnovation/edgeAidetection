import cv2

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open camera. Try index 1 or check /dev/video*.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read a frame")
        break

    cv2.imshow("USB camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
