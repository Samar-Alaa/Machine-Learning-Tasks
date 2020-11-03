"""
Name: Samar Alaa Sobhy
Sec: 1      BN:42
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

mnist =  pd.read_csv("MNIST.csv")

mnist_labels = mnist["label"]
mnist_data = mnist.drop(['label'], axis=1)

train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist_data, mnist_labels, test_size=1/5.0, random_state=0)


sgd_clf = SGDClassifier(max_iter=1000, tol=0.001, random_state = 42)

def digit_3_classification():
    
    train_lbl_3 = (train_lbl == 3)
    test_lbl_3 = (test_lbl == 3)

    sgd_clf.fit(train_img, train_lbl_3)

    prediction_digit_3 = sgd_clf.predict(test_img)

    confusionMatrix = confusion_matrix(test_lbl_3, prediction_digit_3)

    print ("1- binary prediction for digit 3 : \n", prediction_digit_3)
    print ("2- Confusion matrix : \n", confusionMatrix)

def all_digit_classification():
    
    sgd_clf.fit(train_img, train_lbl)

    prediction_all =  sgd_clf.predict(test_img)
    
    print ("3- prediction of all digits : \n", prediction_all )


#To show binary prediction of digit 3 for test data with confusion matrix
digit_3_classification()

#To show prediction of all digits for test data
all_digit_classification()
