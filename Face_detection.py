import cv2
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
        ret, frame = cap.read()  # Read frame continuously
        if not ret:  # Check if frame is read successfully
            print("No frames captured. Exiting...")
            break

        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face = face_classifier.detectMultiScale(
            gframe, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # When outputting a video
        out.write(frame)

        # When outputting a frame
        '''
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show() 
        '''
        # You can add code here to save the frame or process it further
    cap.release()  # Release the video capture object after processing
    out.release()  # Release the output object after processing


# Load the video and convert it to grayscale
video_path = "./Seizure_videos/seizuremp4.mp4"
output_path = "./Seizure_videos/tracking_output.mp4"

face_tracking(video_path, output_path)