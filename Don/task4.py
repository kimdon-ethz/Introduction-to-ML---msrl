import pandas as pd
import numpy as np
import glob
#from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# import the data file
df = np.loadtxt('train_triplets.txt', dtype='str')
dt = np.loadtxt('test_triplets.txt', dtype='str')

def permutation(f):
    p = [0,2,1]
    y = []
    for i in range(0, int(f.shape[0]/2)):
        y = np.append(y, [1], axis=0)
    for i in range(int(f.shape[0]/2), f.shape[0]):
        f[i,:] = f[i,p]
        y = np.append(y, [0], axis=0)
    return f, y.astype('int')

df, y = permutation(df)
X_t, X_v, y_train, y_val = train_test_split(df, y, test_size = 0.2, \
                                                  random_state = 42)

def min_shape(f):
    dd = glob.glob('food/*.jpg')
    m = []
    for i in dd:
#        imgs = Image.open(i)
        imgs = load_img(i)
        m.append(imgs)
        imgs.close()
    n = sorted([(np.sum(i.size), i.size) for i in m])[0][1]
    return n

m = min_shape(df)


def imgtoarr(f, m):
    X = []
    for i in range(0, f.shape[0]):
        a = 'food/' + f[i,0]+'.jpg'
        b = 'food/' + f[i,1]+'.jpg'
        c = 'food/' + f[i,2]+'.jpg'
        row = []
        for k in [a,b,c]:
#            imgs = Image.open(k)
            imgs = load_img(k)
            row.append(imgs)   
        data = np.hstack((np.asarray(row[0].resize(m)), \
                          np.asarray(row[1].resize(m)), \
                          np.asarray(row[2].resize(m))))
        X.append(data)
    return X

X_train = np.array(imgtoarr(df, m))
X_val = np.array(imgtoarr(X_v, m))
X_test = np.array(imgtoarr(dt, m))

#test = Image.fromarray(X_train[0])
#test.show()
#------------------------
#yy_train = y_train[0:10000]
#yy_val = y_val[0:10000]
#yy_train = to_categorical(yy_train, 2)
#yy_val = to_categorical(yy_val, 2)
#
#input_shape = X_train[0].shape
#bsize = 128
#epochs = 50
#
#
#CNN = Sequential()
#CNN.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
#CNN.add(Conv2D(64, (3,3), activation='relu'))
#CNN.add(MaxPooling2D(pool_size=(3,3)))
#CNN.add(Dropout(0.25))
#CNN.add(Flatten())
#CNN.add(Dense(128, activation='relu'))
#CNN.add(Dropout(0.5))
#CNN.add(Dense(2, activation='softmax'))
#
#CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#CNN.fit(X_train, yy_train, batch_size=bsize, epocahs=epochs, verbose=1, \
#        validation_data=(X_val, yy_val))
#
#y_pred = CNN.predict(X_val)
#acc = accuracy_score(yy_val, y_pred)
#print(acc)



#Predict test dataset and make csv output file
#y_test = pd.DataFrame((nn.predict_proba(test_oh)[:,1] >= 0.431).astype('int'))
#y_test.to_csv('result.csv', index=False, header=False)

