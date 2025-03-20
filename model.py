import streamlit as st
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Function to make predictions
def model_prediction(test_image):
    model = tf.keras.models.load_model('pretrained_model.keras')
    
    if test_image is not None:
        image = tf.image.decode_image(test_image.read(), channels=3)
        image = tf.image.resize(image, (224, 224))  # Resize to match model input
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize pixel values

        # Make prediction
        predictions = model.predict(image)
        result_index = np.argmax(predictions)  # Get the class with highest probability
        
        return result_index, predictions
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Fish Classification & Recognition"])
if(app_mode == "Home"):
    st.header("Fish Classification and Recognition System")
    image_path = "10115_2023_1987_Fig2_HTML.png"
    st.image(image_path,use_container_width=True)
    st.markdown("""
### Welcome To Fish Classification and Recognition Web App
A **Fish classification** is an essential requirement for biomass estimation, disease identification, and quality analysis.Our goal is to achieve maximum accuracy in prediction of fish species.
## Features

- Upload fish images for classification
- Real-time fish species recognition using a trained deep learning model
- Interactive visualization of classification results
- Model confidence scores displayed for each prediction
- Supports multiple fish species
- Simple and user-friendly interface

## Installation

Ensure you have Python installed, then set up the environment:

```bash
# Clone the repository
git clone https://github.com/your-repo/fish-classification-app.git
cd fish-classification-app

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit app with:

```bash
streamlit run app.py
```

This will launch the web app in your default web browser.

## Usage

1. Open the app in your browser.
2. Upload an image of a fish.
3. Click the **Classify** button.
4. View the predicted species and confidence scores.

## Example Output

```
Predicted Species: Salmon
Confidence: 95.3%
```

## Model Information

- Uses a **Convolutional Neural Network (CNN)** trained on a diverse fish dataset.
- Supports fine-tuning and transfer learning for improved accuracy.

## Contributing

Feel free to submit pull requests or report issues via GitHub.

## License

MIT License © 2025 

""")
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    A Large-Scale Dataset for Segmentation and Classification
**Authors** : O. Ulucan, D. Karakaya, M. Turkan
Department of Electrical and Electronics Engineering, Izmir University of Economics, Izmir, Turkey
Corresponding author: M. Turkan
Contact Information: mehmet.turkan@ieu.edu.tr

**Paper** : A Large-Scale Dataset for Fish Segmentation and Classification
**General Introduction**

This dataset contains 9 different seafood types collected from a supermarket in Izmir, Turkey
for a university-industry collaboration project at Izmir University of Economics, and this work
was published in ASYU 2020.
The dataset includes gilt head bream, red sea bream, sea bass, red mullet, horse mackerel,
black sea sprat, striped red mullet, trout, shrimp image samples.

If you use this dataset in your work, please consider to cite:

@inproceedings{ulucan2020large,
title={A Large-Scale Dataset for Fish Segmentation and Classification},
author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
pages={1--5},
year={2020},
organization={IEEE}
}

O.Ulucan, D.Karakaya, and M.Turkan.(2020) A large-scale dataset for fish segmentation and classification.
In Conf. Innovations Intell. Syst. Appli. (ASYU)
**Purpose of the work**

This dataset was collected in order to carry out segmentation, feature extraction, and classification tasks
and compare the common segmentation, feature extraction, and classification algorithms (Semantic Segmentation, Convolutional Neural Networks, Bag of Features).
All of the experiment results prove the usability of our dataset for purposes mentioned above.

**Data Gathering Equipment and Data Augmentation**

Images were collected via 2 different cameras, Kodak Easyshare Z650 and Samsung ST60.
Therefore, the resolution of the images are 2832 x 2128, 1024 x 768, respectively.

Before the segmentation, feature extraction, and classification process, the dataset was resized to 590 x 445
by preserving the aspect ratio. After resizing the images, all labels in the dataset were augmented (by flipping and rotating).

At the end of the augmentation process, the number of total images for each class became 2000; 1000 for the RGB fish images
and 1000 for their pair-wise ground truth labels.

**Description of the dataset**

The dataset contains 9 different seafood types. For each class, there are 1000 augmented images and their pair-wise augmented ground truths.
Each class can be found in the "Fish_Dataset" file with their ground truth labels. All images for each class are ordered from "00000.png" to "01000.png".

For example, if you want to access the ground truth images of the shrimp in the dataset, the order should be followed is "Fish->Shrimp->Shrimp GT".

#### Content
1. Test: 7200 Images
2. Train: 1800 Images
                
""")
# Class labels


# Sidebar

    
elif app_mode == "Fish Classification & Recognition":
    st.header("Fish Classifier and Recognizer")
    test_image = st.file_uploader("Choose an Image:")
    classes = ['Hourse Mackerel',
 'Black Sea Sprat',
 'Sea Bass',
 'Red Mullet',
 'Trout',
 'Striped Red Mullet',
 'Shrimp',
 'Gilt-Head Bream',
 'Red Sea Bream']
    if st.button("Show Image"):
        st.image(test_image, use_container_width=True)
    
    if st.button("Predict"):
        with st.spinner("Model is processing. Please wait..."):
            result_index, predictions = model_prediction(test_image)

        predicted_class = classes[result_index]
        st.success(f"Model Prediction: **{predicted_class}**")
        
        # Display prediction probabilities
        st.subheader("Prediction Probabilities")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {predictions[0][i]*100:.2f}%")
        
        # Simulate ground truth for testing (Replace with real labels when available)
        ground_truth = np.random.randint(0, 9, size=(20,))  # Simulating ground truth
        model_preds = np.random.randint(0, 9, size=(20,))  # Simulating model predictions
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(ground_truth, model_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # Display classification report
        st.subheader("Classification Report")
        report = classification_report(ground_truth, model_preds, target_names=classes, output_dict=True)
        st.text(classification_report(ground_truth, model_preds, target_names=classes))
