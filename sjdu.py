import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')
import librosa
import sklearn

filename_list = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
features,labels = np.empty((0,40,44)),np.empty(0)
for i in range(30):
    filePath = "D:\\ccf2020\\%s" % filename_list[i]
    fl = os.listdir(filePath)
    print(len(fl))
    for j in range(len(fl)):
        wavpath = filePath + '\\' + fl[j]
        sig,sr = librosa.load(wavpath)
        #不够长度的信号进行补0
        x = np.pad(sig,(0,22050-sig.shape[0]),'constant')
        #print(x.shape)
        #提取信号的mfcc并进行标准化
        a = librosa.feature.mfcc(x,sr,n_mfcc=40)
        norm_mfccs = sklearn.preprocessing.scale(a,axis=1)
        features = np.append(features,norm_mfccs[None],axis=0)
        labels = np.append(labels,int(i))
        if j < 2:
            print(features.shape)
    print('*****%s*****'%i)

features1 = np.empty((0,28,44))
for i in range(30):
    filePath = "D:\\ccf2020\\%s" % filename_list[i]
    fl = os.listdir(filePath)
    print(len(fl))
    for j in range(len(fl)):
        wavpath = filePath + '\\' + fl[j]
        x,sr = librosa.load(wavpath)
        #不够长度的信号进行补0
        sig = np.pad(x,(0,22050-x.shape[0]),'constant')
        a = librosa.feature.zero_crossing_rate(sig,sr)
        b = librosa.feature.spectral_centroid(sig,sr=sr)[0]
        a = np.vstack((a,b))
        b = librosa.feature.chroma_stft(sig,sr)
        a = np.vstack((a,b))
        b = librosa.feature.spectral_contrast(sig,sr)
        a = np.vstack((a,b))
        b = librosa.feature.spectral_bandwidth(sig,sr)
        a = np.vstack((a,b))
        b = librosa.feature.tonnetz(sig,sr)
        a = np.vstack((a,b))
        norm_a = sklearn.preprocessing.scale(a,axis=1)
        #print(norm_mfccs.shape)
        features1 = np.append(features1,norm_a[None],axis=0)
        if j < 2:
            print(features1.shape)
    print('*****%s*****'%i)
X_train= np.concatenate((features,features1),axis=1)
print(X_train.shape)

#测试集
X_test1 = np.empty((0,40,44))
filePath = "D:\\ccf\\test"
fl = os.listdir(filePath)
print(len(fl))
for j in range(len(fl)):
    wavpath = filePath + '\\' + fl[j]
    sig,sr = librosa.load(wavpath)
    #不够长度的信号进行补0
    x = np.pad(sig,(0,22050-sig.shape[0]),'constant')
    #print(x.shape)
    #提取信号的mfc并进行标准化
    mfcc = librosa.feature.mfcc(x,sr,n_mfcc=40)
    norm_mfccs = sklearn.preprocessing.scale(mfcc,axis=1)
    X_test1 = np.append(X_test1,norm_mfccs[None],axis=0)
    if j < 2:
        print(X_test1.shape)

X_test2 = np.empty((0,28,44))
filePath = "D:\\ccf\\test"
fl = os.listdir(filePath)
print(len(fl))
for j in range(len(fl)):
    wavpath = filePath + '\\' + fl[j]
    x,sr = librosa.load(wavpath)
    #不够长度的信号进行补0
    sig = np.pad(x,(0,22050-x.shape[0]),'constant')
    a = librosa.feature.zero_crossing_rate(sig,sr)
    b = librosa.feature.spectral_centroid(sig,sr=sr)[0]
    a = np.vstack((a,b))
    b = librosa.feature.chroma_stft(sig,sr)
    a = np.vstack((a,b))
    b = librosa.feature.spectral_contrast(sig,sr)
    a = np.vstack((a,b))
    b = librosa.feature.spectral_bandwidth(sig,sr)
    a = np.vstack((a,b))
    b = librosa.feature.tonnetz(sig,sr)
    a = np.vstack((a,b))
    norm_a = sklearn.preprocessing.scale(a,axis=1)
    #print(norm_mfccs.shape)
    X_test2 = np.append(X_test2,norm_a[None],axis=0)
    if j < 2:
        print(X_test2.shape)
X_test = np.concatenate((X_test1,X_test2),axis=1)
print(X_test.shape)

X_train_c = np.array(X_train)
print(X_train_c.shape)
y1_train_c = np.array(labels)
print(y1_train_c.shape)
X_test_c = np.array(X_test)
print(X_test_c.shape)
fileHandle = open ( 'traindata_v2.txt', 'wb+' )
pickle.dump(X_train_c, fileHandle)
fileHandle.close()
fileHandle = open ( 'ydata_v2.txt', 'wb+' )
pickle.dump(y1_train_c, fileHandle)
fileHandle.close()
fileHandle = open ( 'testdata_v2.txt', 'wb+' )
pickle.dump(X_test_c, fileHandle)
fileHandle.close()
