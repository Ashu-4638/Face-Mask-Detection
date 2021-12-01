import cv2 as cv


class Camera:


    def __init__(self):
        self.camera = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.roi_color = 0

        if not self.camera.isOpened():
            raise  ValueError("Unable to Open Camera!")

        self.width = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            ret, image = self.camera.read()
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.face_cascade = cv.CascadeClassifier("E:/Face Mask Detection/haarcascade_frontalface_default.xml")
            self.faces = self.face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5,
                                                            minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in self.faces:
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.roi_color = image[y:y + h, x:x + w]
            cv.imwrite('frame.jpg', self.roi_color)



            if ret:
                return (ret, cv.cvtColor(image, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return None