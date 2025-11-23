import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Load base model
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze base layers initially
base_model.trainable = False

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation="softmax")(x)

# Complete model
resnet_model = Model(inputs=base_model.input, outputs=predictions)

# Compile
resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

resnet_model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ModelCheckpoint("best_resnet_model.keras", save_best_only=True, monitor="val_accuracy")
]

history = resnet_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)
# Unfreeze last few layers of ResNet50
for layer in base_model.layers[-30:]:
    layer.trainable = True

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = resnet_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)
