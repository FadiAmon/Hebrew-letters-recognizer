# Hebrew-letters-recognizer
In this project, a KNN algorithm was created from a given dataset of handwritten images with Hebrew letters to match the images to their respective letters.


Authors:

1) Fadi Amon, EMAIL (contact info): fadiam@ac.sce.ac.il
2) Rasheed Abu Mdegem, EMAIL (contact info): rasheab1@ac.sce.ac.il

Description:

We made a python script in PyCharm environment which trains a ML model to train on hand written images of letters in Hebrew, the images were imported from HHD data set.
The model was made using KNN classifer from sklearn, the model recieves images of letters as np arrays and produces a prediction of the letter to each image.

Environment:

We used windows developing this script, all you need to run it is python and the needed libraries such as, OpenCV, sklearn, PIL, numpy and pandas.


Instructions on how to run the program:

In order to run the program you need to go to the terminal and type the name of the .py file and pass to it 1 argument of the data set path.

Example:
*The data set in this exmaple is in the same directory as HW1.py, hence no full path was written.

python knn_classifier.py ./hhd_dataset 
