from nets.yolo3 import yolo_body
from tensorflow.python.keras.layers import Input
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
yolo = YOLO()

while True:
    cap = cv2.VideoCapture("02.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        print(frame.shape)
        frame = cv2.resize(frame, (960, 500))
        cv2.imwrite("tmp.jpg",frame)
        try:
            image = Image.open("tmp.jpg")
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.save("r_image.jpg")
            eyes = cv2.CascadeClassifier("eye.xml")
            r_image = cv2.imread("r_image.jpg")
            gray_frame = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)
            eye = eyes.detectMultiScale(gray_frame, 1.3, 5)
            tmp_middle =0
            count=0
            for (a, b, c, d) in eye:
                tmp_middle+=(b+(d/2))
                count+=1

            if count!=0:
                tmp_middle /= count
            for (a, b, c, d) in eye:
                # we have to draw the rectangle on the
                # coloured face
                if abs((b + (d / 2)) - tmp_middle) < 35:
                    cv2.rectangle(r_image, (a, b), (a + c, b + d), (0, 255, 0), thickness=4)

            cv2.imshow("new", r_image)
        k = cv2.waitKey(50)
        # q键退出
        if (k & 0xff == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

yolo.close_session()
