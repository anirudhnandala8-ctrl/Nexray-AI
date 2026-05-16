import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#Load dataset with Data Augmentation
#This checks by variations like tilt and zoom
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1)
train_data = train_datagen.flow_from_directory(
    'dataset/train', 
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

#Build a robust model
model = Sequential() 

# first block: focus edges/ribs
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

# Second block: Focuses on lung textures
model.add(Conv2D(64,(3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

#Brain layer: 256 neurons for deeper understanding
model.add(Dense(256, activation='relu'))
#for reduce overfitting
model.add(Dropout(0.5))
#final decision layer
model.add(Dense(1, activation='sigmoid'))

#compile model(learning rules)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
#Train model
model.fit(
    train_data,
    epochs=10,
    verbose=2
)

model.save("xray_model.h5")
print("Model saved successfully")



print(train_data.class_indices)