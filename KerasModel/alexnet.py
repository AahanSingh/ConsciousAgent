from keras.models import Sequential
from keras.layers import Convolution2D,Dense,MaxPooling2D, BatchNormalization, Activation, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img


def dataGen(path):
    # ADDING TRANSFORMATIONS TO SMALL IMAGE DATASET

    print path
    data = ImageDataGenerator(rotation_range=40,        # VALUES WITHIN WHICH TO RANDOMLY ROTATE IMAGES
                              width_shift_range=0.2,    # VALUES  RANDOMLY      IMAGES
                              height_shift_range=0.2,   #       TO         SHIFT
                              rescale=1./255,           # VALUES TO RANDOMLY RESIZE IMAGES
                              shear_range=0.2,          # VALUES TO RANDOMLY SHEAR IMAGES
                              zoom_range=0.2,           # VALUES TO RANDOMLY ZOOM INTO IMAGES
                              horizontal_flip=True,     # RANDOMLY FLIP IMAGES HORIZONTALLY
                              fill_mode='nearest')      # FILLING STRATEGY AFTER SHIFTING OR ROTATION

    img = load_img("/Users/aahansingh/Images/Train/Helicopters/helicopter_1.jpg")            # LOADING IMAGE
    x = img_to_array(img)           # CONVERTING TO NUMERIC ARRAY. SHAPE IS (HEIGHT, WIDTH, CHANNELS)
    x = x.reshape((1,) + x.shape)   # RESHAPING (1, CHANNELS, WIDTH, HEIGHT)
    i = 0

    # GENERATING TRANSFORMED IMAGES IN BATCHES OF 1 AND SAVING TO DIRECTORY NAMES 'PREVIEW'
    for batch in data.flow(x,  batch_size = 1,
                           save_to_dir = 'preview',
                           save_prefix = 'cat',
                           save_format = 'jpeg'):
        i+=1
        if i>20:
            break

#dataGen("abc")
def createModel():
    model = Sequential()
    model.add(Convolution2D(50,5,5,input_shape=(150,150,3)))
    model.add(Activation('relu'))
    model.add((MaxPooling2D(pool_size=(2,2))))

    model.add(Convolution2D(100,5,5))
    model.add(Activation('relu'))
    model.add((MaxPooling2D(pool_size=(2, 2))))

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print 'Build done'
    return model

def trainData():
    data = ImageDataGenerator()
    train = data.flow_from_directory('DataSet/Train', target_size=(150,150), batch_size=10, class_mode='binary')

    model = createModel()
    model.fit_generator(train, samples_per_epoch=5, nb_epoch=4, verbose=1)
    model.save_weights('first_try.h5')



trainData()
model = createModel()
model.load_weights('first_try.h5')
test = ImageDataGenerator().flow_from_directory('DataSet/Test', target_size=(150, 150), batch_size=10, class_mode=None, shuffle=True)
preds = model.predict_generator(test,61)
print preds
