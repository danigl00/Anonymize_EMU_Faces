import cv2
import matplotlib.pyplot as plt

# Load the image
image_cv2_yunet = cv2.imread("./Seizure_videos/futurs_numeriques_2024.png")
height, width, _ = image_cv2_yunet.shape

# Create the face detector
detector = cv2.FaceDetectorYN.create("./Yunet/Models/face_detection_yunet_2023mar.onnx", "", (0, 0))
detector.setInputSize((width, height))

# Detect faces
_, faces = detector.detect(image_cv2_yunet)

# If faces are detected, draw bounding boxes and confidence scores
if faces is not None: 
    for face in faces:
        # parameters: x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
        
        # bounding box
        x, y, w, h = list(map(int, face[:4]))
        color = (0, 0, 255)
        cv2.rectangle(image_cv2_yunet, (x, y), (x+w, y+h), color, 5)
        
        # confidence
        confidence = face[-1]
        confidence_text = "{:.2f}".format(confidence)
        position = (x, y - 10)
        cv2.putText(image_cv2_yunet, confidence_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3, cv2.LINE_AA)

# Convert the image from BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image_cv2_yunet, cv2.COLOR_BGR2RGB)

# Plot the image
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
