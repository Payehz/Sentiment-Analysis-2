![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Sentiment Analysis

The model is created to predict the type of category of a sentiment base on the words used in the sentence.

## Results

The model manages to score an accuracy of 79%. Although it is overfitting, but the model can still be used.

![Accuracy](static/classification_report.png)

The graph of epochs was obtained using TensorBoard

![Val_acc](static/val_acc.png)
![Val_loss](static/val_loss.png)

## Model Architecture

The model used to train this data consists of embedding, bidirectional, spatialdropout1D, LSTM and dense layers.
Each hidden layers contains 128 nodes, after each layer it goes through dropout value of 20%.

![architech](static/model_architechture.png)

## Credits

The data can be obtained from [GitHub](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv)
