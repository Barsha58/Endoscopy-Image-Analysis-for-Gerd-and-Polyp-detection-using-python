from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved .keras model
resnet50 = load_model("/kaggle/working/resnet50_model.keras")

# Class labels
class_labels = ['Gerd', 'Polyp', 'Polyp Normal', 'Gerd Normal']

# Images to test
img_paths = [
    '/kaggle/input/gerdimage/gerd.jpg',
    '/kaggle/input/polypimage/polyp.jpg'
]

plt.figure(figsize=(12, 6))

for i, img_path in enumerate(img_paths):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    pred = resnet50.predict(img_batch)
    pred_idx = np.argmax(pred)
    pred_label = class_labels[pred_idx]
    confidence = np.max(pred)

    print(f"Image: {img_path}")
    print(f"Predicted Class: {pred_label}, Confidence: {confidence:.2f}")

    plt.subplot(1, len(img_paths), i+1)
    plt.imshow(np.array(img).astype('uint8'))
    plt.title(f'{pred_label} ({confidence:.2f})')
    plt.axis('off')

plt.tight_layout()
plt.show()

