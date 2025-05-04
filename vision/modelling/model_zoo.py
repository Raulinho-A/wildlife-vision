import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, optimizers

def build_baseline_model(img_height, img_width, num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
