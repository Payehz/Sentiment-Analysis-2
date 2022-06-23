# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:25:19 2022

@author: User
"""

import matplotlib.pyplot as plt
from tensorflow.keras import Input
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D

class ModelCreation():
    def __init__(self):
        pass
    
    def nlp_model(self,num_feature,num_class,vocab_size,embedding_dim,
                  drop_rate=0.3,num_node=128):
        model = Sequential()
        model.add(Input(shape=(num_feature)))
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(SpatialDropout1D(0.4))
        model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node))
        model.add(Dropout(drop_rate))
        model.add(Dense(num_node,activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(num_class,activation='softmax'))
        
        return model
    
    def result_plot(self,tr_loss,val_loss,tr_acc,val_acc):
        
        plt.figure()
        plt.plot(tr_loss)
        plt.plot(val_loss)
        plt.legend(['train_loss', 'val_loss'])
        plt.show()

        plt.figure()
        plt.plot(tr_acc)
        plt.plot(val_acc)
        plt.legend(['train_acc', 'val_acc'])
        plt.show()

class ModelEvaluation():
    def __init__(self):
        pass
    
    def scores(self,y_true,y_pred):
        
        cm = confusion_matrix(y_true,y_pred)
        cr = classification_report(y_true,y_pred)
        acc_score = accuracy_score(y_true,y_pred)
        print(cm)
        print(cr)
        print(acc_score)















