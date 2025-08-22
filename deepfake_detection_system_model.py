#Imports for future proofing
import os, sys

#Needed Imports
import numpy as np #Needed for works over arrays, collections etc. and type casting etc.
import seaborn as sb #Needed for creating visual representation of data like confusion matrix
import matplotlib.pyplot as plt #Needed for plotting confusion matrix, and other graphs used to evaluate the model's behavior, progress etc.
from sklearn.metrics import confusion_matrix, classification_report #For plotting confusion matrix and saving model
import tensorflow as tf #To use AUTOTUNE to automatically decide buffer_size
from tensorflow.keras import layers, models #For making model
from tensorflow.keras.preprocessing import image_dataset_from_directory #To create dataset from directory
from tensorflow.keras.callbacks import EarlyStopping #To ensure that only needed amount of epochs are run, and the best weights are ensured
import joblib #For saving the model

#Setting up the dataset
dataset_directory = "dataset/train"

#Prerequisites for splitting the dataset
image_height = 128
image_width = 128
batch_size = 32

#Splitting dataset for different purposes,
train_ds_og = image_dataset_from_directory(
    dataset_directory,
    validation_split = 0.2,
    subset = "training",
    seed = 42,
    image_size = (image_height, image_width),
    batch_size = batch_size
)

val_ds_og = image_dataset_from_directory(
    dataset_directory,
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    image_size = (image_height, image_width),
    batch_size = batch_size
)

#Prerequisite for shuffling the datasets, and to automatically chose the buffer_size
AUTOTUNE = tf.data.AUTOTUNE

#Shuffling the datasets for each epoch, also preparing the next batch to speed process up
#Assiging the shuffled dataset to a new one, to ensure that we've a not shuffled version of the datasets ready if needed
train_ds = train_ds_og.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds_og.cache().prefetch(buffer_size = AUTOTUNE)

rescaled_layer = layers.Rescaling(1. / 255) #Rescaling layers so that pixels may have only 0 and 1 as values

#Definition of the CNN model
model = models.Sequential([
    rescaled_layer,

    #Convolution layer 1
    layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (image_height, image_width, 3)),
    layers.MaxPooling2D(2 ,2),

    #Convolution layer 2
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.MaxPooling2D(2, 2),

    #Convolution layer 3
    layers.Conv2D(128, (3, 3), activation = "relu"),
    layers.MaxPooling2D(2, 2),


    layers.Flatten(),

    #Making a neural network of 128 neurons, each to focus on a particular range on input
    layers.Dense(128, activation = "relu"),

    #Dropping Half neurons (randomly) to ensure the model doesn't rely heavily on a single neuron
    layers.Dropout(0.5),

    #Densing the neural network to one single neuron, using sigmoid activation to calculate the probability in range 0 to 1
    layers.Dense(1, activation = "sigmoid")
])

model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

#To stop the epochs once val_loss becomes stagnant, checks for 3 turns and restores the best observed weights
early_stop_epochs = EarlyStopping(
    monitor = "val_loss",
    patience = 3,
    restore_best_weights = True
)

epochs = 50 #Declairing epochs outside to easily update it throughout the code, if needed

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs,
    callbacks = [early_stop_epochs]
)

#Validating accuracy
accuracy, loss = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy * 100:.2f}")

#Making the main figure on which graphs are to be plotted
plt.figure(figsize = (12, 5))

#Making the subplot of Accuracy over epochs
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend() #To put the xlabel and y label in a box
plt.title("Accuracy changes over Epochs")

#Making the subplot of Loss over epochs
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss changes over Epochs")

#Classification data
class_names = train_ds_og.class_names
print("Classification names: ", class_names)

#Showing the graphs
plt.show()

#Prerequisites (declaration) for confusion matrix
y_true = []
y_pred = []

for images, labels in val_ds_og: #Using Unshuffled dataset to ensure processes done on shuffled datasets don't impact the confusion matrix
    preds = model.predict(images)
    preds = (preds > 0.5).astype("int32") #To ensure values >= 0.5 become 1 and the rest zero, in the predictions
    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

#Completing the creation of y_true and y_pred
y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred) #Making the confusion matrix (counts)

#Plotting the confusion matrix (counts)
plt.figure(figsize= (6, 4))
sb.heatmap(cm, annot = True, fmt = "d", cmap = "Reds",
           xticklabels = class_names,
           yticklabels = class_names)
plt.xlabel("Prediction Tabel")
plt.ylabel("Real Tabel")
plt.title("Confusion matrix (counts)")
plt.show()

cm_percentage = cm.astype("float") / cm.sum(axis = 1)[:, np.newaxis] #Making the confusion matrix (percentages)

#Plotting the confusion matrix (percentages)
plt.figure(figsize= (6, 4))
sb.heatmap(cm_percentage, annot = True, fmt = ".2f", cmap = "Reds",
           xticklabels = class_names,
           yticklabels = class_names)
plt.xlabel("Prediction Tabel")
plt.ylabel("Real Tabel")
plt.title("Confusion matrix (percentages)")
plt.show()

class_report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n", class_report)

#Saving the model in .h5 file
model.save("deepfake_detection_system_model.h5")

#Saving the model in .joblib
joblib.dump({
    "model_path": "deepfake_detection_system_model.h5",
    "class_names": class_names
}, "deepfake_detection_system_model.joblib")

print("Model Successfully saved in files:\ndeepfake_detection_system_model.h5\ndeepfake_detection_system_model.joblib")