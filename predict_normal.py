# -----------------------------
#   USAGE
# -----------------------------
# python predict_normal.py --image pig.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.utils import get_class_idx
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import argparse
import imutils
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def preprocess_image(image):
    """
        Swaps color channels, preprocesses the input image and adds a batch dimension on the input image
        :param image: input image
        :return: preprocessed image
    """
    # Swap color channels, preprocesses the image and add in a batch dimension
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    # Return the preprocessed image
    return image


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

# Load the image from disk and make a clone for annotation
print("[INFO] Loading image...")
image = cv2.imread(args["image"])
output = image.copy()

# Preprocess the input image
output = imutils.resize(output, width=400)
preprocessedImage = preprocess_image(image)

# Load the pre-trained ResNet50 model
print("[INFO] Loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")

# Make predictions on the input image and parse the top-3 predictions
print("[INFO] Making predictions...")
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions, top=3)[0]

# Loop over the top three predictions
for (i, (imagenetID, label, prob)) in enumerate(predictions):
    # Print the ImageNet class label ID of the top prediction in the terminal
    # (This class label ID is going to be used in the next script which perform the actual adversarial attack)
    if i == 0:
        print("[INFO] {} => {}".format(label, get_class_idx(label)))
    # Display the prediction to the screen
    print("[INFO] {}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# Draw the top-most predicted label on the image along with the confidence score
text = "{}: {:.2f}%".format(predictions[0][1], predictions[0][2] * 100)
cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)

