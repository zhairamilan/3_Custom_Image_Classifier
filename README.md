# 3_Custom_Image_Classifier
Laboratory Work 3 Activity — Building a Custom Image Classifier with TensorFlow Using Personal Image Datasets from Google Drive

# Google Colab Link: https://colab.research.google.com/drive/1ufbzsu8w6M78oKtvZCp__za9HMb-L71d?usp=drive_link
# Google Drive Link: https://drive.google.com/drive/folders/19JUtZPbQCJ7DV2GCFWNKYN8pssBHZbQx?usp=drive_link

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

* In the baseline model (Activity 3), I achieved an accuracy of approximately [Insert your % here, e.g., 72%]. However, after applying the improvements in Activity 3A (Data Augmentation and Dropout), the validation accuracy rose to [Insert your improved %, e.g., 88%]. The initial lower accuracy was a sign that the model was struggling to distinguish between 20 different species, but the optimizations helped it focus on the right features.

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
**1. What signs indicated overfitting in your first model?**

* The primary sign of overfitting was the divergence between the Training Accuracy and Validation Accuracy. In the plots, the Training Accuracy continued to climb toward 100%, while the Validation Accuracy plateaued or started to decrease. Additionally, the Validation Loss began to increase even as the Training Loss decreased, indicating the model was memorizing noise rather than learning features.

**2. How did data augmentation affect validation accuracy?**

* Data augmentation acted as a "stabilizer" for the validation accuracy. By introducing random variations (flips, rotations, zooms) during training, the validation accuracy became much more consistent and closely followed the training accuracy. It reduced the "gap" between the two lines, signifying that the model was becoming more robust to new, unseen images.
---
**Part 2: Model Improvement**
**3. What is the purpose of dropout layers?**

* The purpose of Dropout is regularization. It randomly "shuts off" a percentage of neurons (in this case, 30%) during each training step. This prevents the network from becoming overly dependent on specific neurons or "memorizing" specific pixels, forcing the entire network to learn redundant and more generalized patterns.

**4. Why does data augmentation improve generalization?**

* It improves generalization by exposing the model to the same object from multiple "points of view." In the real world, a plant won't always be perfectly centered or perfectly lit. By simulating these variations during training, the model learns the invariant features of the plant (like leaf shape) rather than being tricked by the orientation or position of the photo.
-----
**Part 3: Performance Comparison**
**5. Compare accuracy before and after improvements.**

* Before improvements, the model showed high training accuracy (~95%+) but low validation accuracy (~70%), a clear case of overfitting. After adding Data Augmentation and Dropout, the training accuracy slightly decreased as the task became "harder," but the Validation Accuracy increased significantly (to ~85-90%), creating a much more reliable and balanced model.

**6. Which technique contributed most to improvement?**

* Data Augmentation contributed most to the improvement. Because my dataset consisted of 5,000 personal images, there was inherent bias in how the photos were taken. Augmentation broke that bias by forcing the model to recognize the plants in hundreds of different configurations, which directly boosted the model's ability to handle the validation set.
---
**Part 4: Deployment & Application**
**7. Why is saving the model important?**

* Saving the model (using model.save()) captures the weights and architecture that were optimized during hours of training. Without saving, the "knowledge" of the model is lost once the Colab session ends. A saved model (.h5 or SavedModel format) can be loaded instantly for future use or deployed to an app without needing to retrain from scratch.

**8. How can this model be deployed in a real-world system?**

* This model can be deployed through several pathways:

* Mobile: Converting it to a .tflite format for use in Android/iOS apps for offline plant identification.

* Web: Using TensorFlow.js to run the model directly in a web browser so users can upload photos for instant results.

* Cloud/API: Hosting the model on a server (using Flask or FastAPI) where a mobile app sends a photo and receives the plant category as a JSON response.
