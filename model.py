
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from skimage import data, io
from skimage.feature import Cascade

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('f1.jpg')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)
dup = []
for k in prediction:
    for i in k:
        dup.append(i)

print (max(dup))
if max(dup) == prediction[0][0]:
  print("Nikhil")
elif max(dup) == prediction[0][1]:
  print("Kavya")
elif max(dup) == prediction[0][2]:
  print("Nithya")
elif max(dup) == prediction[0][3]:
  print("Nikhil V Gopal")
else:
  print("Unknown")
image.show()