# BOVW
Image classification using bag of visual words,SVM and RF
# Abstract
Bag of visual words is image classification approach that is inspired of bag of word that is used for document classification.In this assignment this technique is applied on two datasets. This report will explain image classification pipe and result obtain from this approach. SIFT extractor and  descriptor is used to extract features, clustering is used for code book construction and for classification SVM and Random Forrest is used.
![CV_daigram](https://user-images.githubusercontent.com/99320378/224541319-7f9cbc97-036b-4f84-9ac8-ba966c1f7367.PNG)


# Instructions for setting up the environment for running your code

This code is can run google colabtory and link is provided in notebook.

If code is to be run on local environment  

1) Code is written in python 3.
2) opencv,matplotlib,slearn,random,shutil are library used.

 
# Instructions on running the train and test scripts on the train and test data
If data is already splited in test and train set it should be in flowing manner

train

''''Class 1

''''Class 2

''''Class 3

test

''''Class 1

''''Class 2

''''Class 3

If data is one folder it should be in subfolder and name of folder represent class. A code block is add in model2 nootbook to takecare . If data is splited the notebook in model1 is better to use. 

# Quantitative results
**Object dataset**
```
SVM Performance:
              precision    recall  f1-score   support

 Soccer_Ball       1.00      1.00      1.00         2
   accordian       1.00      1.00      1.00         2
 dollar_bill       1.00      1.00      1.00         2
   motorbike       1.00      1.00      1.00         2

    accuracy                           1.00         8
   macro avg       1.00      1.00      1.00         8
weighted avg       1.00      1.00      1.00         8

Class 1:
True Positives (TP): 2
False Positives (FP): 0
True Negatives (TN): 6
False Negatives (FN): 0
True Positive Rate (TPR): 1.0000
False Positive Rate (FPR): 0.0000
Class 2:
True Positives (TP): 2
False Positives (FP): 0
True Negatives (TN): 6
False Negatives (FN): 0
True Positive Rate (TPR): 1.0000
False Positive Rate (FPR): 0.0000
Class 3:
True Positives (TP): 2
False Positives (FP): 0
True Negatives (TN): 6
False Negatives (FN): 0
True Positive Rate (TPR): 1.0000
False Positive Rate (FPR): 0.0000
Class 4:
True Positives (TP): 2
False Positives (FP): 0
True Negatives (TN): 6
False Negatives (FN): 0
True Positive Rate (TPR): 1.0000
False Positive Rate (FPR): 0.0000
F1 Score:  1.0
Accuracy:  1.0
```
```
Random Forest Performance:
              precision    recall  f1-score   support

 Soccer_Ball       1.00      1.00      1.00         2
   accordian       1.00      0.50      0.67         2
 dollar_bill       1.00      1.00      1.00         2
   motorbike       0.67      1.00      0.80         2

    accuracy                           0.88         8
   macro avg       0.92      0.88      0.87         8
weighted avg       0.92      0.88      0.87         8

Class 1:
True Positives (TP): 2
False Positives (FP): 0
True Negatives (TN): 6
False Negatives (FN): 0
True Positive Rate (TPR): 1.0000
False Positive Rate (FPR): 0.0000
Class 2:
True Positives (TP): 1
False Positives (FP): 0
True Negatives (TN): 6
False Negatives (FN): 1
True Positive Rate (TPR): 0.5000
False Positive Rate (FPR): 0.0000
Class 3:
True Positives (TP): 2
False Positives (FP): 0
True Negatives (TN): 6
False Negatives (FN): 0
True Positive Rate (TPR): 1.0000
False Positive Rate (FPR): 0.0000
Class 4:
True Positives (TP): 2
False Positives (FP): 1
True Negatives (TN): 5
False Negatives (FN): 0
True Positive Rate (TPR): 1.0000
False Positive Rate (FPR): 0.1667
F1 Score:  0.8666666666666667
Accuracy:  0.875 
```
**FlowersData**
```
SVM Performance:
              precision    recall  f1-score   support

       daisy       0.65      0.51      0.57       126
   dandelion       0.58      0.79      0.67       179
       roses       0.58      0.43      0.49       128
  sunflowers       0.62      0.71      0.66       139
      tulips       0.50      0.43      0.46       159

    accuracy                           0.58       731
   macro avg       0.59      0.57      0.57       731
weighted avg       0.58      0.58      0.57       731

Class 1:
True Positives (TP): 64
False Positives (FP): 35
True Negatives (TN): 570
False Negatives (FN): 62
True Positive Rate (TPR): 0.5079
False Positive Rate (FPR): 0.0579
Class 2:
True Positives (TP): 141
False Positives (FP): 103
True Negatives (TN): 449
False Negatives (FN): 38
True Positive Rate (TPR): 0.7877
False Positive Rate (FPR): 0.1866
Class 3:
True Positives (TP): 55
False Positives (FP): 40
True Negatives (TN): 563
False Negatives (FN): 73
True Positive Rate (TPR): 0.4297
False Positive Rate (FPR): 0.0663
Class 4:
True Positives (TP): 98
False Positives (FP): 59
True Negatives (TN): 533
False Negatives (FN): 41
True Positive Rate (TPR): 0.7050
False Positive Rate (FPR): 0.0997
Class 5:
True Positives (TP): 68
False Positives (FP): 68
True Negatives (TN): 504
False Negatives (FN): 91
True Positive Rate (TPR): 0.4277
False Positive Rate (FPR): 0.1189
F1 Score:  0.573863997595134
Accuracy:  0.5827633378932968
```

```
Random Forest Performance:
              precision    recall  f1-score   support

       daisy       0.52      0.43      0.47       126
   dandelion       0.51      0.70      0.59       179
       roses       0.42      0.30      0.35       128
  sunflowers       0.59      0.61      0.60       139
      tulips       0.50      0.47      0.48       159

    accuracy                           0.51       731
   macro avg       0.51      0.50      0.50       731
weighted avg       0.51      0.51      0.51       731

Class 1:
True Positives (TP): 54
False Positives (FP): 50
True Negatives (TN): 555
False Negatives (FN): 72
True Positive Rate (TPR): 0.4286
False Positive Rate (FPR): 0.0826
Class 2:
True Positives (TP): 125
False Positives (FP): 119
True Negatives (TN): 433
False Negatives (FN): 54
True Positive Rate (TPR): 0.6983
False Positive Rate (FPR): 0.2156
Class 3:
True Positives (TP): 38
False Positives (FP): 52
True Negatives (TN): 551
False Negatives (FN): 90
True Positive Rate (TPR): 0.2969
False Positive Rate (FPR): 0.0862
Class 4:
True Positives (TP): 85
False Positives (FP): 60
True Negatives (TN): 532
False Negatives (FN): 54
True Positive Rate (TPR): 0.6115
False Positive Rate (FPR): 0.1014
Class 5:
True Positives (TP): 74
False Positives (FP): 74
True Negatives (TN): 498
False Negatives (FN): 85
True Positive Rate (TPR): 0.4654
False Positive Rate (FPR): 0.1294
F1 Score:  0.5053853952495385
Accuracy:  0.5143638850889193
```
# Visual Results
**Object dataset**

![2](https://user-images.githubusercontent.com/99320378/224541284-b08ea426-559e-4039-b11e-a19b744e1f9d.png)

**Flower Dataset**

![3](https://user-images.githubusercontent.com/99320378/224541296-c78a3b2d-5b8b-40cf-bc81-be1dff7506ae.png)
