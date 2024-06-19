import cv2
import numpy as np
from tqdm import tqdm

def face_tracking(video_path, 
                  output_path, 
                  threshold, 
                  model,
                  color, 
                  box_thickness, 
                  radius, 
                  landmarks_thickness, 
                  text_scale, 
                  text_thickness, 
                  font
                  ):
    
    capture = cv2.VideoCapture(video_path)
    model_path = f"./Yunet/Models/{model}.onnx" #face_detection_yunet_2023mar.onnx
    #rewrite = cv2.VideoCapture("./Seizure_videos/yunet_tracking_output2.mp4")

    face_detector = cv2.FaceDetectorYN_create(model_path, "", (0, 0), threshold) 

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = tqdm(total=frame_count, desc = "Processing frames", unit = "frames")

    while True:
        ret, image = capture.read()
        #ret, image_2 = rewrite.read()
        if not ret:
            print("No frames captured. Exiting...")
            break

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            box = list(map(int, face[:4]))
            cv2.rectangle(image, box, color, box_thickness)
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                cv2.circle(image, landmark, radius, color, landmarks_thickness, cv2.LINE_AA)             
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10) 
            cv2.putText(image, confidence, position, font, text_scale, color, text_thickness, cv2.LINE_AA)
        bar.update(1)
        # Display the processed frame
        cv2.putText(image, model, (10, 70), font, text_scale, color, text_thickness)
        output.write(image)

    capture.release()
    output.release()
    bar.close()



face_tracking(
    video_path = "./Seizure_videos/seizuremp4.mp4",
    output_path = "./Seizure_videos/yunet_tracking_output2.mp4",
    threshold = 0.67,
    model = "yunet_n_640_640", #face_detection_yunet_2023mar
    color = (0, 255, 0),
    box_thickness = 2,
    radius = 3,
    landmarks_thickness = -1,
    text_scale = 1,
    text_thickness = 1,
    font = cv2.FONT_HERSHEY_PLAIN
)
