import face_recognition
import numpy as np
import cv2

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
nikhil_image = face_recognition.load_image_file("Nikhil.jpg")
nikhil_face_encoding = face_recognition.face_encodings(nikhil_image)[0]

# Load a second sample picture and learn how to recognize it.
kavya_image = face_recognition.load_image_file("Kavya.jpg")
kavya_face_encoding = face_recognition.face_encodings(kavya_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    nikhil_face_encoding,
    kavya_face_encoding
]
known_face_names = [
    "Nikhil",
    "Kavya"
]

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
