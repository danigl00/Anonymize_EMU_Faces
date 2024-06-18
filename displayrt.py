import cv2
import matplotlib.pyplot as plt

face_classifier1 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_classifier2 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)

face_classifier3 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

face_classifier4 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt_tree.xml"
)

face_classifier5 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

def face_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frames captured. Exiting...")
            break
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # CLASSIFIER 1
        face = face_classifier1.detectMultiScale(
            gframe, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "default", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            
        # Display the processed frame
        cv2.imshow("Face Tracking", frame)

        # Use a key press to exit (e.g., 'q' or 'Esc')
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or 'Esc'
            break

    cap.release()
    cv2.destroyAllWindows()


# Load the video
video_path = "./Seizure_videos/seizuremp4.mp4"

face_tracking(video_path)
