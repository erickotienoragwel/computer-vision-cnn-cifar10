# 🧠 CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

## 📘 Project Overview
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow and Keras** to classify images from the **CIFAR-10 dataset**.  
The model is capable of recognizing 10 different object categories such as **airplanes, automobiles, birds, cats, and dogs**, achieving over **70% test accuracy** after 10 training epochs.

This project is part of the **AI Engineering Beginner Series Project Implementation**, where learners apply their foundational deep learning knowledge to solve real-world computer vision problems.

---

## 🎯 Project Objectives
1. Build a CNN model to classify images into predefined categories.  
2. Learn how to preprocess and normalize image data for training.  
3. Understand how to compile, train, and evaluate deep learning models.  
4. Apply the model to predict unseen images and interpret confidence scores.  
5. Demonstrate practical understanding of image recognition using AI.

---

## 🧰 Step-by-Step Implementation

### Step 1: Import Libraries
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

### Step 2: Load and Prepare the CIFAR-10 Dataset
```python
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0
```

### Step 3: Build the CNN Model
```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
model.summary()
```

### Step 4: Compile and Train the Model
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
```

### Step 5: Evaluate Model Performance
```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\n✅ Test accuracy: {test_acc:.2f}')
```

### Step 6: Predict New Images
```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def predict_new_image(model, img_path):
    img = Image.open(img_path).resize((32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob_model = Sequential([model, Softmax()])
    prediction = prob_model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    print(f"🖼️ Image Path: {img_path}")
    print(f"🔍 Predicted Class: {class_names[predicted_class]}")
    print(f"✅ Confidence: {confidence*100:.2f}%")

    plt.imshow(img)
    plt.title(f"{class_names[predicted_class]} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()
```

---

## 📊 Results and Performance
After training for 10 epochs, the model achieved:
- **Training Accuracy:** 78.17%  
- **Validation Accuracy:** 70.32%  
- **Loss:** 0.89  

The CNN successfully recognized several classes such as **truck**, **airplane**, **horse**, **frog**, and **bird** with high confidence (above 90%).

---

## 💡 Application Areas
This project demonstrates foundational skills applicable in various AI and computer vision applications:

| Application Area | Description |
|------------------|-------------|
| 🏭 **Autonomous Vehicles** | Used for object and traffic sign recognition. |
| 📱 **Mobile Apps** | Powers image filters, recognition, and real-time object detection. |
| 🏥 **Healthcare** | Adapted for classifying X-ray, MRI, and CT scan images. |
| 🛒 **E-commerce** | Enables visual search and product image classification. |
| 🌍 **Environmental Monitoring** | Recognizes wildlife species or pollution patterns using satellite images. |
| 🧑‍🏫 **Education** | Used to teach deep learning fundamentals and applied AI development. |

---

## 🚀 How to Run the Project

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Matplotlib
- Pillow

### Run the Notebook
1. Clone this repository or copy the code into a Colab notebook.  
2. Run all cells in order.  
3. Use your own images in the `predict_new_image()` function for testing.

---

## 📦 Submission Details
- **Course:** AI Engineering Beginner Series  
- **Module:** Project Implementation  
- **Instructor:** Martial School of IT  
- **Submission Portal:** [AI Engineering Project Implementation Portal](https://jengaelearning.com/courses/ai-engineering-beginner-series-project-implementation/)  
- **Deliverables:**
  - Jupyter Notebook / Python Script (`.ipynb` or `.py`)
  - README.md (Project Documentation)
  - Trained Model (optional)
  - Screenshots of predictions (optional)

---

## 🧑‍💻 Author
**Erick Otieno**  
AI Trainer, Martial Jenga Labs
📍 Nairobi, Kenya  
💼 Passionate about empowering youth through technology and AI innovation.

---

## 🏁 Conclusion
This project provides an excellent foundation for understanding how CNNs work in real-world image recognition tasks. Learners are encouraged to experiment with **data augmentation**, **dropout layers**, or **transfer learning** (e.g., using VGG16 or ResNet) to improve accuracy further.
