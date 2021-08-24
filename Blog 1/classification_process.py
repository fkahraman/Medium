
import os, cv2
from glob import glob

import numpy as np
from tensorflow.keras.utils import to_categorical

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

from keras.engine.saving import model_from_json

def resize_image():

    # List of files
    dir_path = r'train/train/'
    file_list = glob(f"{dir_path}*")

    # Create folder
    if not os.path.exists('Reize_train'):
        os.makedirs('Reize_train')

    for index, file_path in enumerate(file_list):

        # Extract filename and file count
        file_name = file_path.split('\\')[-1]
        label = file_name.split('.')[0]
        count = file_name.split('.')[1]

        record_name = f"{label}.{count.zfill(5)}.resized.jpg"

        # Resize operations
        img = cv2.imread(file_path)
        resized_image = cv2.resize(img, (64,64))
        cv2.imwrite(f"Reize_train/{record_name}", resized_image)

        # Show info screen
        if index % 100 == 0:
            print(f"Completed file count {index} | {record_name}")


def create_data():

    # Per label digitization limit
    limit = 1000

    # List of files
    dir_path = r'Reize_train/'
    file_list = glob(f"{dir_path}*")

    # Create folder
    if not os.path.exists('Data'):
        os.makedirs('Data')

    all_data_feature = []
    all_label = []

    for index, file_path in enumerate(file_list):

        # Extract filename and file count
        file_name = file_path.split('\\')[-1]
        label = file_name.split('.')[0]
        count = file_name.split('.')[1]

        if int(count) >= limit:
            continue

        # Convert img to array with opencv
        img = cv2.imread(file_path)

        all_data_feature.append(img)

        # Labeling
        if label == 'cat':
            all_label.append(0)
        else:
            all_label.append(1)

    # Convert list to numpy array
    all_data_feature_npy = np.asarray(all_data_feature).astype('float32')
    all_label_npy = np.asarray(all_label).astype('float32')

    # Normalization
    all_data_feature_npy /= 255

    # Show old label format
    print('\n',all_label_npy[0:10],'\n')

    # Change to categorical format
    all_label_npy = to_categorical(all_label_npy)

    # Show new label format
    print(all_label_npy[0:10],'\n')

    # Show shapes
    print(all_data_feature_npy.shape,'\n')
    print(all_label_npy.shape,'\n')

    # Record numpy format
    np.save('Data/train_X.npy', all_data_feature_npy)
    np.save('Data/train_y.npy', all_label_npy)


def get_model():

    model = Sequential()

    # Block 1
    model.add(Conv2D(16, (3, 3), input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.summary()

    return model

def train(model, train_X, train_y):

    # Train process
    model.fit(train_X, train_y, validation_split=0.33,  epochs=30, batch_size=1, verbose=True)

    # Create folder
    if not os.path.exists('Model'):
        os.makedirs('Model')

    # Convert json format
    model_json = model.to_json()

    # Record json
    with open(f"Model/model_json.json", "w") as model_file:
        model_file.write(model_json)

    # Save model
    model.save_weights(f"Model/final_model.h5")

    return

def prediction():

    # Model json load
    model_file = open('Model/model_json.json', 'r')
    model = model_file.read()
    model_file.close()

    # Model weights load
    model = model_from_json(model)
    model.load_weights('Model/final_model.h5')

    # Read test images
    test_1 = cv2.imread('Reize_train/cat.05000.resized.jpg')
    test_2 = cv2.imread('Reize_train/cat.07000.resized.jpg')
    test_3 = cv2.imread('Reize_train/dog.03000.resized.jpg')
    test_4 = cv2.imread('Reize_train/cat.06000.resized.jpg')

    print(f"Read form | {test_1.shape}")

    # Convert correct form
    test_1 = np.expand_dims(test_1, axis=0)
    test_2 = np.expand_dims(test_2, axis=0)
    test_3 = np.expand_dims(test_3, axis=0)
    test_4 = np.expand_dims(test_4, axis=0)

    print(f"Correct form | {test_1.shape}\n")

    prediction_list = []

    # Predictions
    predict_1 = model.predict(test_1)
    predict_2 = model.predict(test_2)
    predict_3 = model.predict(test_3)
    predict_4 = model.predict(test_4)

    # Collection predictions
    prediction_list.append(predict_1)
    prediction_list.append(predict_2)
    prediction_list.append(predict_3)
    prediction_list.append(predict_4)

    # Convert array to label
    for pred in prediction_list:

        if pred[0][0] == 1:
            print("Cat")
        else:
            print("Dog")

if __name__ == '__main__':

    resize_image()
    create_data()

    model = get_model()

    train_X = np.load('Data/train_X.npy')
    train_y = np.load('Data/train_y.npy')

    train(model, train_X, train_y)

    prediction()