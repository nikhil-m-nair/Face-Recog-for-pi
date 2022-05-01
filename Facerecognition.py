import face_recognition
from PIL import Image, ImageDraw,ImageFont
import numpy as np

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

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("f1.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)


pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Dundee"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(48, 63, 159))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(48, 63, 159), outline=(48, 63, 159))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 0))


# Remove the drawing library from memory as per the Pillow docs
del draw

pil_image.show()