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

parser = argparse.ArgumentParser(description='')
parser.add_argument('data_set_path', type=str)
args = parser.parse_args()

data_set_path=args.data_set_path



def pad(image):

    width,length=image.shape[0:2]
    white = [255,255,255]
    padded_image=image
    if width < length:
        # top, bottom, left, right

        if length-width==1:
            padding=1
            #print("here")
            padded_image = cv2.copyMakeBorder(image,padding,0,0,0, cv2.BORDER_CONSTANT, value = white)
        else:

            #print("here1")
            if length%2==1:
                padding = (length + 1 - width)//2
                add=1
            else:
                padding = (length - width)//2
                add=0
            padded_image = cv2.copyMakeBorder(image,padding,padding,add,0, cv2.BORDER_CONSTANT, value = white)

    elif length < width:
        # top, bottom, left, right
        #print("here 2")
        if width-length==1:
            padding=1
            padded_image = cv2.copyMakeBorder(image,padding,padding,0,0, cv2.BORDER_CONSTANT, value = white)
        else:
            #print("here 3")
            if width%2==1:
                padding = (width + 1 - length)//2
                add=1
            else:
                padding = (width - length)//2
                add=0

            padded_image = cv2.copyMakeBorder(image,add,0,padding,padding, cv2.BORDER_CONSTANT, value = white)

    return padded_image


def pre_proccessing(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image=pad(image)
    image = cv2.resize(image, dsize=(32,32))
    img_binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    return img_binary




base_dir = fr'{data_set_path}'
#Read all the folders in the directory
folder_list = os.listdir(base_dir)
print( len(folder_list), "categories found in the dataset")

batch_x=[]
batch_y=[]
index=0

for image_class in (folder_list):
    images_in_class = os.listdir(os.path.join(base_dir, image_class))
    results_for_class=[]
    for image in images_in_class: #loop over each image in a certain class
        image=os.path.join(base_dir, fr'{image_class}/', image)
        image=pre_proccessing(image)
        batch_x.append(image)
        batch_y.append(int(image_class))
        index+=1
batch_x=np.array(batch_x)


X_train, X_test, y_train, y_test = train_test_split(batch_x, batch_y, test_size=0.1, random_state=1)
len(X_train),len(X_test)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1112087912, random_state=1)
len(X_train),len(X_val)


len(X_train),len(y_train),len(X_val),len(y_val),len(X_test),len(y_test)

def reshape_data(images):
    nsamples, nx, ny = images.shape
    data = images.reshape((nsamples,nx*ny))
    return data

X_train_train_dataset=reshape_data(X_train)
X_train_train_dataset.shape


neigh = KNeighborsClassifier(n_neighbors=1,p=2)
neigh.fit(X_train_train_dataset, y_train)


val_data=reshape_data(X_val)
neigh.score(val_data,y_val)


test_data=reshape_data(X_test)
y_test=np.array(y_test)
neigh.score(test_data,y_test)
y_pred = neigh.predict(test_data)



classes_images_dict={}
base_dir = fr'{data_set_path}'
#Read all the folders in the directory
folder_list = os.listdir(base_dir)
print( len(folder_list), "categories found in the dataset")

batch_x=[]

for image_class in (folder_list):
    batch_x=[]
    images_in_class = os.listdir(os.path.join(base_dir, image_class))
    results_for_class=[]
    for image in images_in_class: #loop over each image in a certain class
        image=os.path.join(base_dir, fr'{image_class}/', image)
        image=pre_proccessing(image)
        batch_x.append(image)
    batch_x=np.array(batch_x)
    data_images=reshape_data(batch_x)
    classes_images_dict[int(image_class)] = neigh.predict(data_images)

classes_acc_dict={}
for letter_class in sorted(list(classes_images_dict.keys())):
    predicted_labels=classes_images_dict[letter_class]
    class_acc=accuracy_score([letter_class]*len(classes_images_dict[letter_class]), predicted_labels)
    classes_acc_dict[letter_class]=class_acc


def save_results_to_txt(k, classes_acc_dict):
    with open('results.txt', 'w') as f:
        f.write(f'k={k}\n')
        f.write('Letter\t\tAccuracy\n')
        for letter_class in list(classes_acc_dict.keys()):
            f.write(f'{letter_class}\t\t{classes_acc_dict[letter_class]}\n')


save_results_to_txt(1, classes_acc_dict)


classes = [*range(27)]
data=[]
cm=(confusion_matrix(y_test, y_pred,labels=neigh.classes_))

for i in cm:
    data.append(i)

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=classes,index=classes)
df.to_csv('confusion_matrix.csv', sep='\t', encoding='utf-8') 
