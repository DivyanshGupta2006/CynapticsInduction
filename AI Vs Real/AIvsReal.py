import os
import keras
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.src.utils import load_img, to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm
from keras.src.callbacks import EarlyStopping
from tqdm.notebook import tqdm

earlystopping = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=2)
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

def extract_features(images):
    features = []
    for image in images:
        try:
            img = load_img(image, target_size=(236, 236))  # Resize
            '''
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.convert_to_tensor(img)  # Convert numpy array to TensorFlow tensor

        # Perform TensorFlow-specific operations
            img = tf.image.random_crop(img, size=[236, 236, 3])  # Random crop
            img = tf.image.resize(img, (236, 236))  # Resize back to 236x236
            '''
            img = np.array(img)  # Convert to numpy array
            features.append(img)
        except:
            pass

    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features

def scheduler(epoch, lr):
    if epoch>2:
        return lr*0.9
    else:
        return lr

TRAIN_DIR = "C:\\Users\\aryam\\OneDrive\\Desktop\\Data\\Train"

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

train_features = extract_features(train['image'])
x_train = train_features / 255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)

model = Sequential()
# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.fit(x=x_train, y=y_train, batch_size=25, epochs=10, callbacks=[lr_scheduler,earlystopping])

def create_test_data_frame(dir):
    image_paths=[]
    for imageName in os.listdir(dir):
        image_paths.append(os.path.join(dir, imageName))
    return image_paths

Test_dir="C:\\Users\\aryam\\OneDrive\\Desktop\\Data\\Test"

test = pd.DataFrame()
test['image']=create_test_data_frame(Test_dir)
print(test['image'])
'''
index = test[test['image'] == "kaggle/Data/Test\image_62.jpg"].index
test = test.drop(index).reset_index(drop=True)
'''
test_features= extract_features(test['image'])
x_test = test_features/255.0
prediction=model.predict(x_test)
predicted_classes = np.argmax(prediction, axis=1)

results = pd.DataFrame({
       "Id": [os.path.basename(path) for path in test['image']],
       "Label": predicted_classes
})
results['Id'] = results['Id'].str.replace('.jpg', '')
results['NumericId'] = results['Id'].str.extract(r'(\d+)').astype(int)
results = results.sort_values(by='NumericId').drop(columns=['NumericId'])
results['Label'] = results['Label'].replace({0: 'AI', 1: 'Real'})
output_csv="res.csv"
results.to_csv(output_csv, index=False)
