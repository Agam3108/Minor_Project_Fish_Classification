# Fish_Classification_Recognition_Minor
 
#🐟 Fish Classification and Recognition System
📌 Introduction to the Problem
Estimating the relative abundance of fish species in their habitats and monitoring population fluctuations are crucial for marine scientists and conservationists. Traditionally, this has been done through manual sampling, which is time-consuming and inefficient.

To tackle this, automated computer-based fish sampling has been introduced using underwater images and videos. However, accurate fish detection and classification remain a challenge due to:

🌊 Environmental fluctuations (lighting changes, water clarity)

🐠 Fish camouflage

🎥 Dynamic backgrounds

🖼️ Low-resolution images

🔄 Shape deformations of moving fish

🧐 Tiny differences between some species

This project aims to provide an efficient and accurate solution to classify fish species using deep learning. Users can upload an image of a fish and get the predicted species name along with model confidence scores.

📂 Understanding the Dataset
This dataset contains 9 different seafood types for classification:
✅ Gilt-head bream
✅ Red sea bream
✅ Sea bass
✅ Red mullet
✅ Horse mackerel
✅ Black sea sprat
✅ Striped red mullet
✅ Trout
✅ Shrimp

For each class, there are 1000 augmented images and their pair-wise augmented ground truth labels.

📌 Dataset Organization:

All images for each class are stored in the Fish_Dataset folder.

Ground truth labels are stored in a structured format.

Images are ordered from "00000.png" to "01000.png".

📌 Example:
If you want to access the ground truth images of shrimp, the order should be followed as:
Fish -> Shrimp -> Shrimp GT
🧠 Model Information - MobileNetV2 Architecture
We have used MobileNetV2, a lightweight deep learning model, for fish classification.

Why MobileNetV2?
⚡ Optimized for Mobile & Web – Low computational cost

🎯 High Accuracy – Uses depthwise separable convolutions

🚀 Efficient & Fast – Works well on low-powered devices

MobileNetV2 Architecture
MobileNetV2 is a convolutional neural network (CNN) architecture designed for efficient classification, particularly on mobile devices. It is based on an inverted residual structure where the residual connections exist between bottleneck layers.

Key Features of MobileNetV2:
✅ Initial convolution layer with 32 filters
✅ 19 residual bottleneck layers
✅ Lightweight depthwise convolutions for feature extraction
✅ Non-linear activation functions for better classification

This model is pre-trained on large datasets and then fine-tuned on the fish classification dataset for improved accuracy.

🚀 About the Project
Our Fish Classification and Recognition Web App allows users to:
✅ Upload an image of a fish
✅ Get the predicted species and confidence score
✅ View real-time classification results
✅ Experience a user-friendly interface

We use deep learning models to classify fish species accurately from an input image.

⚙️ Installation & Setup
Ensure you have Python installed, then set up the environment:

bash
Copy
Edit
# Clone the repository
git clone https://github.com/Agam3108/Minor_Project_Fish_Classification.git
cd Minor_Project_Fish_Classification

# Install dependencies
pip install -r requirements.txt
▶️ Running the Application
Run the Streamlit web app with:
streamlit run model.py
This will launch the web application in your default browser.

📝 Usage
1️⃣ Open the app in your browser.
2️⃣ Upload an image of a fish.
3️⃣ Click the Predict button.
4️⃣ View the predicted species and confidence score.

Example Output
Predicted Species: Salmon  
Confidence: 95.3%
