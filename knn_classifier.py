import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import argparse

# Initialize argument parser to allow user to specify data set path
parser = argparse.ArgumentParser(description='')
parser.add_argument('data_set_path', type=str)
args = parser.parse_args()

data_set_path=args.data_set_path

def pad(image):
    """
    Pads an image so that it is square.

    Args:
        image (np.ndarray): The image to pad.

    Returns:
        np.ndarray: The padded image.
    """
    width, length = image.shape[0:2]
    white = [255, 255, 255]
    padded_image = image
    
    if width < length:
        # Pad the sides with white pixels
        if length - width == 1:
            padding = 1
            padded_image = cv2.copyMakeBorder(
                image, padding, 0, 0, 0, cv2.BORDER_CONSTANT, value=white
            )
        else:
            if length % 2 == 1:
                padding = (length + 1 - width) // 2
                add = 1
            else:
                padding = (length - width) // 2
                add = 0
            padded_image = cv2.copyMakeBorder(
                image, padding, padding, add, 0, cv2.BORDER_CONSTANT, value=white
            )

    elif length < width:
        # Pad the top and bottom with white pixels
        if width - length == 1:
            padding = 1
            padded_image = cv2.copyMakeBorder(
                image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=white
            )
        else:
            if width % 2 == 1:
                padding = (width + 1 - length) // 2
                add = 1
            else:
                padding = (width - length) // 2
                add = 0

            padded_image = cv2.copyMakeBorder(
                image, add, 0, padding, padding, cv2.BORDER_CONSTANT, value=white
            )

    return padded_image


def pre_proccessing(image_path):
    """
    Preprocesses an image by resizing it to 32x32 pixels, converting it to grayscale,
    and applying binary thresholding.

    Args:
        image_path (str): The file path of the image to preprocess.

    Returns:
        np.ndarray: The preprocessed image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = pad(image)
    image = cv2.resize(image, dsize=(32, 32))
    img_binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    return img_binary



# Set the base directory to the dataset path
base_dir = fr'{data_set_path}'

# Read all the folders in the directory
folder_list = os.listdir(base_dir)

# Print the number of categories found in the dataset
print( len(folder_list), "categories found in the dataset")

# Initialize empty lists for the image data and labels
batch_x=[]
batch_y=[]
index=0

# Loop over each category in the dataset
for image_class in (folder_list):
    
    # Get the list of images in the current category
    images_in_class = os.listdir(os.path.join(base_dir, image_class))
    
    # Initialize an empty list for the results for the current category
    results_for_class=[]
    
    # Loop over each image in the current category
    for image in images_in_class: 
        
        # Get the full path of the image and preprocess it
        image=os.path.join(base_dir, fr'{image_class}/', image)
        image=pre_proccessing(image)
        
        # Add the preprocessed image data and label to the batch lists
        batch_x.append(image)
        batch_y.append(int(image_class))
        
        # Increment the index counter
        index+=1

# Convert the batch data to a NumPy array
batch_x=np.array(batch_x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(batch_x, batch_y, test_size=0.1, random_state=1)

# Print the number of training and testing samples
print(len(X_train),len(X_test))

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1112087912, random_state=1)

# Print the number of training, validation, and testing samples
print(len(X_train),len(X_val))
print(len(X_train),len(y_train),len(X_val),len(y_val),len(X_test),len(y_test))

def reshape_data(images):
    """
    Reshape the image data from a 3D tensor to a 2D matrix.
    
    Args:
    images: A NumPy array of image data
    
    Returns:
    A NumPy array of reshaped image data
    """
    nsamples, nx, ny = images.shape
    data = images.reshape((nsamples,nx*ny))
    return data


# Reshape the training data
X_train_train_dataset=reshape_data(X_train)
X_train_train_dataset.shape

# Initialize and fit the KNN classifier with k=1 and p=2
neigh = KNeighborsClassifier(n_neighbors=1,p=2)
neigh.fit(X_train_train_dataset, y_train)

# Reshape the validation data and calculate the accuracy of the classifier on the validation data
val_data=reshape_data(X_val)
neigh.score(val_data,y_val)

# Reshape the test data and calculate the accuracy of the classifier on the test data
test_data=reshape_data(X_test)
y_test=np.array(y_test)
neigh.score(test_data,y_test)
y_pred = neigh.predict(test_data)

# Create a dictionary to store the predicted labels for each image class
classes_images_dict={}

# Read all the folders in the directory and loop over each image in each folder
base_dir = fr'{data_set_path}'
folder_list = os.listdir(base_dir)
print( len(folder_list), "categories found in the dataset")
for image_class in (folder_list):
    batch_x=[]
    images_in_class = os.listdir(os.path.join(base_dir, image_class))
    results_for_class=[]
    for image in images_in_class:
        # Load and preprocess each image
        image=os.path.join(base_dir, fr'{image_class}/', image)
        image=pre_proccessing(image)
        batch_x.append(image)
    batch_x=np.array(batch_x)
    # Reshape the images in the batch
    data_images=reshape_data(batch_x)
    # Predict the labels for the images in the batch using the KNN classifier
    classes_images_dict[int(image_class)] = neigh.predict(data_images)

# Calculate the accuracy for each image class and store the results in a dictionary
classes_acc_dict={}
for letter_class in sorted(list(classes_images_dict.keys())):
    predicted_labels=classes_images_dict[letter_class]
    class_acc=accuracy_score([letter_class]*len(classes_images_dict[letter_class]), predicted_labels)
    classes_acc_dict[letter_class]=class_acc

# Save the results to a text file
def save_results_to_txt(k, classes_acc_dict):
    """
    Save the accuracy results for each class to a text file.
    
    Args:
    - k: int, the number of neighbors used in the KNN classifier
    - classes_acc_dict: dict, a dictionary containing the accuracy results for each class
    
    Returns:
    - None
    """
    # Open the file in write mode
    with open('results.txt', 'w') as f:
        # Write the value of k to the file
        f.write(f'k={k}\n')
        # Write the column headers to the file
        f.write('Letter\t\tAccuracy\n')
        # Loop over each class in the dictionary
        for letter_class in list(classes_acc_dict.keys()):
            # Write the class label and accuracy value to the file
            f.write(f'{letter_class}\t\t{classes_acc_dict[letter_class]}\n')

save_results_to_txt(1, classes_acc_dict)

# Create a confusion matrix and save it to a CSV file
classes = [*range(27)]
data=[]
cm=(confusion_matrix(y_test, y_pred,labels=neigh.classes_))
for i in cm:
    data.append(i)

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=classes,index=classes)
df.to_csv('confusion_matrix.csv', sep='\t', encoding='utf-8')
