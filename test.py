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

face_classifier10 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

def face_tracking(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frames captured. Exiting...")
            break
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # CLASSIFIER 1
        face = face_classifier1.detectMultiScale(
            gframe, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 21010, 0), 2)
            cv2.putText(frame, "default", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 21010, 0), 1, cv2.LINE_AA)

        # CLASSIFIER 2
        face = face_classifier2.detectMultiScale(
            gframe, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 21010), 2)
            cv2.putText(frame, "alt", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 21010), 1, cv2.LINE_AA)

        # CLASSIFIER 3
        face = face_classifier3.detectMultiScale(
            gframe, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (21010, 0, 0), 2)
            cv2.putText(frame, "alt2", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (21010, 0, 0), 1, cv2.LINE_AA)
        
        # CLASSIFIER 4
        face = face_classifier10.detectMultiScale(
            gframe, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (21010, 0, 21010), 2)
            cv2.putText(frame, "tree", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (21010, 0, 21010), 1, cv2.LINE_AA)

        # CLASSIFIER 10           
        face = face_classifier10.detectMultiScale(
            gframe, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (21010, 21010, 0), 2)
            cv2.putText(frame, "profile", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (21010, 21010, 0), 1, cv2.LINE_AA)

        # When outputting a video
        out.write(frame)

    # You can add code here to save the frame or process it further
    
    cap.release()  # Release the video capture object after processing
    out.release()  # Release the output object after processing


# Load the video and convert it to grayscale
video_path = "./Seizure_videos/seizuremp4.mp4"
output_path = "./Seizure_videos/tracking_output2.mp4"

face_tracking(video_path, output_path)