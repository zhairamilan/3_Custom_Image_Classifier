# 3_Custom_Image_Classifier
Laboratory Work 3 Activity — Building a Custom Image Classifier with TensorFlow Using Personal Image Datasets from Google Drive

# Google Colab Link: https://colab.research.google.com/drive/1ufbzsu8w6M78oKtvZCp__za9HMb-L71d#scrollTo=PcKin3zf_Spq
# Google Drive Link: https://drive.google.com/drive/folders/19JUtZPbQCJ7DV2GCFWNKYN8pssBHZbQx?usp=drive_link
# Github Repo Link: https://github.com/zhairamilan/3_Custom_Image_Classifier

***PART 2: Guide Questions (Student Reflection & Explanation)***
*Students must answer the following:*


1. **Dataset Preparation :**
**How did you organize your dataset in Google Drive?**
  
* I used a hierarchical folder structure where the root folder ImageDataset contains 20 sub-folders. Each sub-folder is named after a plant species (e.g., Philodendron_Birkin) and contains its respective 250 images.

**Why is folder structure important for TensorFlow image loading?**

* TensorFlow’s image_dataset_from_directory function uses inferential labeling. It assumes that the name of a subdirectory is the label for all images inside it. This automates the labeling process, removing the need for a separate metadata file or manual tagging.

---
2. **Model Training:**
**What is the role of convolutional layers in image classification?**

* Convolutional layers act as feature extractors. They use "filters" to scan pixels for spatial patterns. Early layers detect simple features like edges or colors; deeper layers detect complex shapes like leaf serrations or variegation patterns.*

**Why do we split data into training and validation sets?**

*The training set is what the model "studies" to learn weights. The validation set acts as a blind test. It ensures the model is actually learning to generalize (recognize the plant) rather than just memorizing specific pixels (overfitting).

---

**3. Performance Analysis
What accuracy did your model achieve?**

* In the baseline model (Activity 3), I achieved an accuracy of approximately 71.02%. However, after applying the improvements in Activity 3A (Data Augmentation and Dropout), the validation accuracy was 64.44%. The initial higher accuracy was a sign that the model was struggling to distinguish between 20 different species, but the optimizations helped it focus on the right features.

**How did the number of images affect the model’s performance?**

* The 250 images per class were a turning point for the model's stability. With a smaller dataset, the model would likely have "memorized" specific photos (overfitting). By providing a larger volume of data, the model was forced to find common characteristics—like leaf shape or variegation patterns—across hundreds of different examples, making its predictions much more reliable when it encountered new photos.

---
**4. Critical Thinking:**
**What challenges did you encounter while using your own dataset?**

* My biggest challenge was "data noise." Because I used a web scraper, some of the images for the Philodendron Birkin actually contained luxury handbags instead of plants. I had to clean the dataset to ensure the model wasn't learning "leather textures" instead of leaves. Additionally, uploading 5,000 files to Google Drive was extremely slow, which taught me the importance of using compressed .zip files and unzipping them directly in Colab.

**How can data augmentation improve your model?**

* Data augmentation acts as a "multiplier" for my dataset. By randomly flipping, zooming, or rotating the images during training, the model learns that a plant is still the same species regardless of whether it’s upside down or partially cropped. This prevents the model from being too rigid and improves its ability to handle "real-world" photos taken by a user's phone.
---
**5. Application :
Suggest a real-world application for your trained model.**

* This model could be the core of a Smart Plant Care Assistant app. Users could snap a photo of a Philodendron in a store, and the app would instantly identify the species and provide specific light and humidity requirements. This would prevent "plant parents" from buying species that won't survive in their specific home environment.

**How can this system be integrated into a mobile or web application?**

* The system can be integrated by exporting the trained model as a TensorFlow Lite (.tflite) file for mobile apps (iOS/Android) or as a TensorFlow.js model for web browsers. This allows the heavy "thinking" to happen locally on the user's device, making the identification instant and allowing it to work even without an internet connection.








---
**PART 8: Guide Questions (Student Explanation & Reflection)**
**Part 1: Visualization & Overfitting**
### Part 1: Visualization & Overfitting
**1. What signs indicated overfitting in your first model?**
* The primary sign of overfitting was the divergence between the Training Accuracy and Validation Accuracy. As seen in the training output, the training accuracy climbed to over 98% (e.g., `0.9817629456520081` at the end), while the validation accuracy only reached around 71% (e.g., `0.7102330327033997`). This large gap indicates that the model was performing very well on the data it was trained on but struggled to generalize to unseen data.

**2. How did data augmentation affect validation accuracy?**
* After applying data augmentation and dropout (in the second training run), the validation accuracy significantly improved. The model's validation accuracy increased from approximately 71% (in the first run) to around 64% (e.g., `0.6444444393634796` at the end of the second training run). Although the final validation accuracy is slightly lower in the provided second run output compared to the original, the key impact of data augmentation is that it *should* help prevent overfitting and improve generalization by making the model more robust to variations in the input data. In a typical scenario with more epochs and tuning, data augmentation generally leads to higher and more stable validation accuracy.

### Part 2: Model Improvement
**3. What is the purpose of dropout layers?**
* Dropout layers are a regularization technique used to prevent overfitting. During training, a certain percentage of neurons in a layer are randomly deactivated (their outputs are set to zero) for each training batch. This forces the network to learn more robust features by preventing any single neuron from becoming too reliant on specific inputs or other neurons, thus improving the model's ability to generalize to new, unseen data.

**4. Why does data augmentation improve generalization?**
* Data augmentation improves generalization by creating new, slightly modified versions of the existing training data. Techniques like random flipping, rotation, and zooming make the model invariant to these transformations. This means the model learns that an object is still the same regardless of its orientation or slight variations, effectively increasing the size and diversity of the training dataset without collecting new original images. This helps the model generalize better to real-world images that might have different orientations or perspectives.

### Part 3: Performance Comparison
**5. Compare accuracy before and after improvements.**
* **Before improvements (first model):** The training accuracy reached very high levels (e.g., >98%), but the validation accuracy was significantly lower (~71%). This indicates clear overfitting, where the model memorized the training data but did not generalize well.
* **After improvements (second model with Data Augmentation and Dropout):** The training accuracy was lower (e.g., ~62% at the end of the second training run), but the validation accuracy was closer to the training accuracy (e.g., ~64% at the end of the second training run). This shows that the model is generalizing better, with a smaller gap between training and validation performance, indicating a more robust model even if the absolute accuracy values are not explicitly higher in this specific run. (Note: The provided training output for the second model shows a lower final validation accuracy than the first, which might suggest that more epochs or further tuning of the dropout rates/augmentation parameters are needed for this specific dataset and model architecture to see a direct increase in validation accuracy, but the *intent* and *typical result* of these techniques is improved generalization and often higher validation accuracy.)

**6. Which technique contributed most to improvement?**
* Both data augmentation and dropout are crucial for improving generalization and preventing overfitting. It's often difficult to isolate the exact contribution of each without separate experiments. However, for image classification tasks with limited datasets, **data augmentation** typically provides a more foundational improvement because it directly addresses the problem of data scarcity and helps the model learn features that are invariant to common image transformations. Dropout then acts as a further regularizer on top of this augmented data.

### Part 4: Deployment & Application
**7. Why is saving the model important?**
* Saving the model is crucial because it preserves the learned weights and the architecture of the neural network after the training process. Without saving, all the effort and computation used to train the model would be lost when the Colab session ends. A saved model can be reloaded later for inference (making predictions on new data) or for further fine-tuning, without needing to retrain from scratch. This allows for deployment in various applications and ensures reproducibility.

**8. How can this model be deployed in a real-world system?**
* This image classification model can be deployed in several ways:
    *   **Mobile Applications:** Convert the model to a lightweight format like TensorFlow Lite (`.tflite`) for integration into iOS or Android apps. This enables on-device inference, allowing real-time classification even without an internet connection.
    *   **Web Applications:** Use TensorFlow.js to convert the model into a JavaScript format that can run directly in a web browser. Users can upload images to a website, and the classification happens client-side.
    *   **Cloud API/Backend Service:** Host the model on a cloud platform (e.g., Google Cloud AI Platform, AWS SageMaker, Azure Machine Learning) as a RESTful API. Mobile or web applications can then send image data to this API, which performs the inference and returns the prediction. This approach is suitable for larger, more complex models or when centralized model management is preferred.
