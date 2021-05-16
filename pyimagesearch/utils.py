# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import json
import os


# -----------------------------
#   FUNCTIONS
# -----------------------------
def get_class_idx(label):
    # Build the path to the ImageNet class label mappings file
    labelPath = os.path.join(os.path.dirname(__file__), "imagenet_class_index.json")
    # Open the ImageNet class mappings file and load the mappings as a dictionary with the human-readable class label
    # as the key and the integer index as the value
    with open(labelPath) as f:
        imageNetClasses = {labels[1]: int(idx) for (idx, labels) in json.load(f).items()}
    # Check to see if the input class label has a corresponding integer index value and if so return it;
    # Otherwise return a None-type value;
    return imageNetClasses.get(label, None)

