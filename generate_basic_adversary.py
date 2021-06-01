# -----------------------------
#   USAGE
# -----------------------------
# python generate_basic_adversary.py --input pig.jpg --output adversarial.png --class-idx 341

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.losses import MSE
import tensorflow as tf
import numpy as np
import argparse
import cv2
import pickle

# -----------------------------
#   FUNCTIONS
# -----------------------------
data_path = 'cifar-100-python/test'
meta_path = 'cifar-100-python/meta'

def load_labels_name(filename):
    
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj



def loading(data_path):

    print("[INFO] Loading cifar100 dataset...")

    train_p = open(data_path, 'rb')
    train = pickle.load(train_p, encoding = "bytes")
    data = {"x_train": train[b'data'], "y_train": train[b'fine_labels']}
    obj_cifar100 = load_labels_name(meta_path)
    data['x_train'] = data['x_train'].reshape((-1, 3, 32, 32))
    data['x_train'] = data['x_train'].T.astype(float)
    data['x_train'] = np.moveaxis(data['x_train'], -1, 0)
    label_names = obj_cifar100['fine_label_names']

    trainX = data['x_train']
    print(trainX.shape)
    trainY = data['y_train']
    print("[INFO] Cifar100 dataset loaded")
#trainX = trainX / 255.0
#testX = testX / 255.0
    return trainX, trainY, label_names
    



def clip_eps(tensor, eps):
    """
        Clips the values of the tensor according to a given range and returns it
        :param tensor: list of tensors
        :param eps: epochs range
        :return: tensor value according to given range
    """
    # Clip the values of the tensor to a given range and return it
    return tf.clip_by_value(tensor, clip_value_min=-eps, clip_value_max=eps)

def generate_image_adversary(model, image, label, eps=2/255.0):
    # Cast the image
    image = tf.cast(image, tf.float32)
    # Record the gradients
    with tf.GradientTape() as tape:
        # Explicitly indicate that the image should be tacked for gradient updates
        tape.watch(image)
        # Use the model to make predictions on the input image and then compute the loss
        pred = model(image)
        loss = MSE(label, pred)
    # Calculate the gradients of loss with respect to the image, then compute the sign of the gradient
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)
    # Construct the image adversary
    adversary = (image + (signedGrad * eps)).numpy()
    # Return the image adversary to the calling function
    return adversary
    

def generate_adversaries(model, baseImage, delta, classIdx, defense, steps=50):

    """
        Generate binaries adversaries for the input image given a base model
        :param model: base model
        :param baseImage: input image
        :param delta: perturbation vector
        :param classIdx: original class index
        :param steps: number of steps
        :return: new perturbation vector
    """
    # Iterate over the number of steps
    for step in range(0, steps):
        # Record the gradients
        with tf.GradientTape() as tape:
            # Explicitly indicate that the perturbation vector should be tracked for gradient updates
            tape.watch(delta)
            # Add the perturbation vector to the base image and preprocess the resulting image
            #adversary = preprocess_input(baseImage + delta)
            adversary = baseImage + delta
            # Run this newly constructed image tensor through the base model and calculate the loss with respect to the
            # original class index
            predictions = model(adversary, defense = defense)

            #print(predictions)
            loss = -sccLoss(tf.convert_to_tensor([classIdx]), predictions)
            # Check the loss value and display it in the terminal
            #if step % 10 == 0:
             #   print("[INFO] Step: {}, Loss: {}...".format(step, loss.numpy()))
        # Calculate the gradients of loss with respect to the perturbation vector
        gradients = tape.gradient(loss, delta)
        # Update the weights, clip the perturbation vector and update its value
        optimizer.apply_gradients([(gradients, delta)])
    delta.assign_add(clip_eps(delta, eps=EPS))
    #deltaImage = delta.numpy().squeeze()
    #deltaImage = cv2.cvtColor(deltaImage, cv2.COLOR_RGB2BGR)
    #cv2.imshow("Output1", deltaImage)
    #print(delta)
    # Return the perturbation vector
    return delta


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--index", type=int, required=True, help="index")
#ap.add_argument("-o", "--output", required=True, help="path to output adversarial image")
ap.add_argument("-s", "--test-size", type=int, required=True, help="how many images you would like to run")
ap.add_argument("-m", "--model", required=True, help="the model you would like to use")

ap.add_argument("-d", "--defense", required=True,help="the defense you would like to test")

args = vars(ap.parse_args())

# Define the epsilon and learning rate constants
EPS = 2 / 255
LR = 0.005

# Load the input image from disk and preprocess it
print("[INFO] Loading image...")
test_size= args["test_size"]

# Load the pre-trained ResNet50 model for running inference
print("[INFO] Loading the pre-trained model...")

#model = tf.keras.models.load_model("final.h5")
model = tf.keras.models.load_model(args["model"])

defense = args["defense"]


# Initialize the optimizer and loss function
optimizer = Adam(learning_rate=LR)
sccLoss = SparseCategoricalCrossentropy()

print("[INFO] Preprocessing...")

trainX, trainY, label_names = loading(data_path)
count = 0


for image_index in np.random.choice(np.arange(0, len(trainX)), size=(test_size,)):

    image = trainX[image_index]
    image = np.expand_dims(image, axis=0)
    
    #predict_index = trainY[image_index]
    predict_index = model.predict(image)
    label = np.argmax(predict_index, axis=1)
    class_name = label_names[int(label)]
   # print("[INFO] predicted label:{} , {}".format(int(label),class_name))

   

    # Create a tensor based of the input image and initialize the perturbation vector that will be updated via training
    baseImage = tf.constant(image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

    # Generate the perturbation vector to create the adversarial example of the base image

    deltaUpdated = generate_adversaries(model, baseImage, delta, defense, int(label))


    # Create the adversarial example, swap color channels and save the output image to disk
   
    # Run inference with this adversarial example, parse the results and display the top-1 predicted results
   
    #preprocessedImage = preprocess_input(baseImage + deltaUpdated)
    adverImage = baseImage +  deltaUpdated
    #predictions = model.predict(preprocessedImage)

    predictions = model(adverImage)

    adver_label = np.argmax(predictions, axis=1)
    class_name = label_names[int(adver_label)]
    adverImage = (baseImage + deltaUpdated).numpy().squeeze()
    #adverImage = np.clip(adverImage, 0, 255).astype("uint8")
    #adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
    
    #print("[INFO] Label after adversary attack: {}, {} ".format(int(adver_label), class_name))
    """
    adversary = generate_image_adversary(model, image, predict_index, eps=0.1)
    predictions2 = model.predict(adversary)
    adver_label2 = np.argmax(predictions2, axis=1)
    class_name2= label_names[int(adver_label2)]
    
    print("[INFO] Label after adversary attack: (fgsm){}, {} ".format(int(adver_label2), class_name2))
    """
    
    if int(label) != int(adver_label):
        count+=1
    
accuracy = count/test_size
print("[INFO] accuracy: {}%".format(accuracy*100))

    # Draw the top-most predicted label on the adversarial image along with the confidence score
    #text = "{}%".format(label)
    #cv2.putText(adverImage, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output image
#cv2.imshow("Output2", adverImage)
cv2.waitKey(0)
