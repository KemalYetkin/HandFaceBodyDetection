#python 3.7 + mediapipe + opencv-python + cvzone + plyer

import cv2, cvzone
from cvzone.PlotModule import LivePlot
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
from plyer import notification
import time

def show_notification_el(message):
    notification.notify(
        title='Dikkat!',
        message=message,
        app_name='El Tespiti',
        timeout=3  # Bildirimin ekranda kalma süresi (saniye cinsinden)
    )



# video yakalamak için webcam i yükler
cap = cv2.VideoCapture(0)

# El ve yüz tanıma class ları tanımlanır ve parametreleri ayarlanır.
hdetector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
fdetector = FaceDetector(minDetectionCon=0.85, modelSelection=0)
pdetector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

xPlot = LivePlot(w=1200, yLimit=[0, 500], interval=0.01)
el_tespit_edildi = False
#yuz_tespit_edildi = False  # Kontrol değişkeni

# sürekli olarak webcam dan görüntü yakalama
while True:
    # kameradan her kareyi yakalaması için
    # görüntü her an yakalandığında 'success' True değerini döndürür ve img değişkenine görüntü aktarılır
    success, img = cap.read()
    # Resize the image to 640x480
    #img = cv2.resize(img, (640, 480))
    # mevcut görüntüdeki elleri bulmak için
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = hdetector.findHands(img, draw=True, flipType=True)
    img, bboxs = fdetector.findFaces(img, draw=False)
    img = pdetector.findPose(img)
    lmList, bboxInfo  = pdetector.findPosition(img, draw=True, bboxWithHands=False)
    if lmList:
        # Extract the center of the bounding box around the detected pose
        center = bboxInfo["center"]

        # Visualize the center of the bounding box
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Calculate the distance between landmarks 11 and 15 and visualize it
        length, img, info = pdetector.findDistance(lmList[11][0:2],
                                                lmList[15][0:2],
                                                img=img,
                                                color=(255, 0, 0),
                                                scale=10)

        # Calculate and visualize the angle formed by landmarks 11, 13, and 15
        # This can be used as an illustrative example of how posture might be inferred from body landmarks.
        angle, img = pdetector.findAngle(lmList[11][0:2],
                                        lmList[13][0:2],
                                        lmList[15][0:2],
                                        img=img,
                                        color=(0, 0, 255),
                                        scale=10)
        # Check if the calculated angle is close to a reference angle of 50 degrees (with a leeway of 10 degrees)
        isCloseAngle50 = pdetector.angleCheck(myAngle=angle,
                                            targetAngle=50,
                                            offset=10)
        # Print the result of the angle comparison
        print(isCloseAngle50)
    # Display the processed frame
    cv2.imshow("Image", img)
    # Introduce a brief pause of 1 millisecond between frames
    cv2.waitKey(1)
    val = 0
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'
            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            val = center[0]
            # ---- Draw Data  ---- #
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{score}%', (x, y - 10))
            cvzone.cornerRect(img, (x, y, w, h))
            # Check if the face detection score is above 90%
            if score > 90:
                notification_text = "Insan Yuzu Tespiti"
                cv2.putText(img, notification_text, (img.shape[1] - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                # Show a notification
#                notification.notify(
#                    title='Yüz Tespiti',
#                    message='İnsan Yüzü Tespit Edildi',
#                    app_icon=None,  # eğer ikon kullanmak istemiyorsanız None bırakabilirsiniz
#                    timeout=10,  # bildirimin kaç saniye boyunca görüntüleneceğini belirtir
#                )
#                # Set the control variable to True to avoid repeated notifications
#                yuz_tespit_edildi = True
    imgPlot = xPlot.update(val)
    cv2.imshow("Image Plot", imgPlot)
    cv2.imshow("Image", img)
    # eğer el tespit edilirse
    if hands:
        # ilk el tespit edildiğindeki bilgi
        if not el_tespit_edildi:
            print("El tespit edildi!")
            show_notification_el('El Tespit Edildi.')
            el_tespit_edildi = True  # Bildirimi bir kere göstermek için
        hand1 = hands[0]  # ilk el tespitini al
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        # Count the number of fingers up for the first hand
        fingers1 = hdetector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

        # Calculate distance between specific landmarks on the first hand and draw it on the image
        length, info, img = hdetector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                    scale=10)
        # Check if a second hand is detected
        if len(hands) == 2:
            # Information for the second hand
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]

            # Count the number of fingers up for the second hand
            fingers2 = hdetector.fingersUp(hand2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

            # Calculate distance between the index fingers of both hands and draw it on the image
            length, info, img = hdetector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                        scale=10)
        print(" ")  # New line for better readability of the printed output

    # Display the image in a window
    cv2.imshow("Image", img)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    # Kullanıcı çıkış yapmak isterse
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break