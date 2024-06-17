import cv2
video = "p302_seizure_anonymized.mp4"
cap = cv2.VideoCapture(video)
ret, frame = cap.read()
cap.release()
cv2.imshow("First Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

