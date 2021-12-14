import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow.lite as tflite


flower_types = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']


def build_model(conv_layers=1, learning_rate=3e-3, dropout_rate=0.2):
    inputs = keras.Input(shape=(target_size[0], target_size[1], 3))

    conv = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    pooling = keras.layers.MaxPool2D(strides=(2, 2))(conv)

    for layer in range(conv_layers):
        conv = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(pooling)
        pooling = keras.layers.MaxPool2D(strides=(2, 2))(conv)

    flatten = keras.layers.Flatten()(pooling)
    dropout = keras.layers.Dropout(rate=dropout_rate)(flatten)
    dense = keras.layers.Dense(64, activation='relu')(dropout)
    outputs = keras.layers.Dense(len(flower_types), activation='softmax')(dense)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    return model


def get_best_model():
    file_names = os.listdir('./')
    models = [m for m in file_names if m.endswith('train.h5')]
    best_model = models[0]
    for model in models:
        if model.split('_')[3] > best_model.split('_')[3]:
            best_model = model
    return best_model


if __name__ == '__main__':
    target_size = (150, 150)
    path = '../data/flowers/'

    train_generator = ImageDataGenerator(rescale=1. / 255)

    train_data = train_generator.flow_from_directory(
        f'{path}train/',
        target_size=target_size,
        batch_size=20
    )

    val_generator = ImageDataGenerator(rescale=1. / 255)

    val_data = val_generator.flow_from_directory(
        f'{path}validation',
        target_size=target_size,
        batch_size=20,
        shuffle=True
    )

    chechpoint = keras.callbacks.ModelCheckpoint(
        'flower_model_{epoch:02d}_{val_accuracy:.3f}_train.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max')

    conv_layers = 2
    learning_rate = 1e-3
    dropout_rate = 0.2

    m = build_model(learning_rate=learning_rate, conv_layers=conv_layers, dropout_rate=dropout_rate)
    history = m.fit(
        train_data,
        steps_per_epoch=100,
        epochs=10,
        validation_data=val_data,
        validation_steps=10,
        callbacks=[chechpoint])

    best_model_path = get_best_model()
    m.load_weights(best_model_path)

    converter = tflite.TFLiteConverter.from_keras_model(m)
    tflite_model = converter.convert()

    model_path = './model.tflite'
    with open(model_path, 'wb') as f_out:
        f_out.write(tflite_model)
