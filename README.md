# Introduction
Floods are one of the most devastating natural disasters that affect millions of people worldwide every year. The ability to quickly and accurately assess the damage caused by floods is crucial for emergency responders, disaster relief organizations, and government agencies to plan and allocate resources effectively. Satellite imagery provides a unique and powerful tool for post-flood damage assessment, as it can cover large areas and capture high-resolution images of the affected regions. However, manually analyzing these images is a time-consuming and labor-intensive process, making it challenging to provide timely and accurate assessments. The goal of this project is to develop an automated method for detecting post-flood damages in satellite imagery using machine learning and computer vision techniques. The proposed solution will allow for the rapid and accurate identification of damaged areas, enabling emergency responders to quickly prioritize their response efforts and allocate resources more effectively. The proposed solution can also be used to monitor the long-term impact of floods on the affected areas, allowing for more informed decision-making and better disaster preparedness in the future. In summary, the proposed solution has the potential to make a significant impact on disaster response and recovery efforts, ultimately improving the lives of people affected by floods.

# Motivation
Detecting post-flood damages in satellite imagery is a challenging yet important task that has real-world applications. The idea behind the project is to leverage machine learning and computer vision techniques to automatically detect and classify the presence of flood damage in satellite images.
There are several dimensions to this project, including:
1- Image preprocessing: This involves preprocessing satellite images of the affected areas to remove any noise, artifacts, or distortion that may affect the accuracy of the analysis. (The images have different dimensions)
2- Feature extraction and selection: In this step, relevant features or patterns are extracted from the preprocessed images using any techniques you find fit. The selected features are then used as input to the classification model.
3- Classification: This step involves training a machine learning model to classify the images as either damaged or undamaged. The model is trained on a dataset of flood images, where there are 2 types of images either flooded or not.
4- Evaluation: Finally, the performance of the model is evaluated using metrics such as accuracy, precision, recall, and F1 score

# Preprocessing
The preprocessing steps for this project involved resizing all images to (512, 512) and performing haze removal on each image using a suitable algorithm. The purpose of these steps was to prepare the images for subsequent analysis and feature extraction. Resizing the images to a common size helps to standardize the dataset so that the images can be compared and analyzed more easily. A size of (512, 512) was chosen based on the specific requirements of the project and the available computing resources. Haze removal is an important step in satellite imaging applications because atmospheric haze can obscure features of interest in the images. There are various algorithms that can be used for haze removal, including dark channel prior, guided filter, and color attenuation prior. The specific algorithm used in this project was haze removal by dark subtraction, which was chosen based on its effectiveness in removing haze from satellite images. The preprocessing steps were implemented using python, and the resulting preprocessed images were saved in a separate directory for further analysis. Overall, these preprocessing steps helped to improve the quality and usability of the satellite images for subsequent analysis and modeling.

# Features Extraction
Local Binary Pattern was used to extract features A Feature vector of length 64 was extracted for each image We used a bigger Radius and No. of points in LBP to extract more global features.

# Classifier
We used two models for the classification task: Pretrained ResNet50 Model & Logistic Regression. For the ResNet50 Model the input was the images itself after preprocessing, resizing to 256*256 and normalization For the Logistic Regression Model we used LBP features extracted as 64 feature vector for each image As expected the ResNet50 Model clearly outperformed Logistic Regression, however Logistic Regression got a very good result.

# Segmentation
Segmentation step or called clustering step is concerned with segment the flooded image into the flooded region and non flooded, that may be used in another step as a caution area, or for rescue squad guidance, .etc.
This step is done using the IsoData clustering algorithm given that number of classes is two.


# Evaluation

| Model             | Confusion Matrix | Precision | Recall | Accuracy | Omission Error | Commission Error | F1 Score |
|-------------------|------------------|-----------|--------|----------|----------------|------------------|----------|
| ResNet50          | 68               | 0.96      | 0.96   | 0.964    | 0.04           | 0.04             | 0.96     |
|                   | 3                | 66        |        |          |                |                  |          |
| Logistic Regression | 58             | 0.82      | 0.82   | 0.82     | 0.18           | 0.18             | 0.82     |
|                   | 13               | 56        |        |          |                |                  |          |
