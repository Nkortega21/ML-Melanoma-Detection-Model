# ML-Skin-Cancer-Detection-Model
## Table of Contents
- [Introduction](#Introduction)
- [Tools](#Tools)
- [Key Steps](#key-steps)
- [References for the Data](#References-for-the-Data)
- [Declaration](#Declaration)

## Introduction
With changes in the environment, UV radiation has become a significant cause of skin cancer. The early symptoms of skin cancer are often difficult to detect with the naked eye, leading many patients to seek medical attention only in the late stages of the disease, missing the optimal treatment window. Traditional diagnosis relies on experienced dermatologists, but their assessments can be limited by the following factors:

Subjectivity: Diagnostic results can vary between doctors due to differences in experience and perspective.
Resource Scarcity: In many remote or under-resourced areas, the number of professional dermatologists is insufficient, resulting in delayed diagnoses.
Meanwhile, machine learning has the potential to efficiently and accurately process large-scale data. Could it be used to improve the speed and efficiency of screening, thereby helping people with early prevention, diagnosis, and reducing unnecessary medical costs and burdens?

This project aims to answer the question: **Can we use machine learning to build a model that determines the likelihood of skin cancer by uploading images of the skin?** This would help people detect skin cancer early and address the shortcomings of traditional medical resources.

## Tools
kaggle，google colab

## Key Steps

### Model Creation Process

#### 1. Data Preparation:
- The dataset is loaded from the specified directories using `ImageDataGenerator` to preprocess images (normalizing pixel values).
- The training data is split into training and validation sets using the `validation_split` parameter.

#### 2. Model Architecture:
- A simple Convolutional Neural Network (CNN) is created using Keras `Sequential` API.
- The model includes:
  - Two convolutional layers (`Conv2D`) with ReLU activation and MaxPooling layers (`MaxPooling2D`) for downsampling.
  - A flattening layer (`Flatten`) to convert the 2D matrix into a 1D vector.
  - Dense layers (`Dense`) for fully connected layers, including an output layer with a sigmoid activation function for binary classification.

#### 3. Model Compilation:

- The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.

#### 4. Model Training:

- The model is trained using the training data, validated on the validation data, and the number of epochs is set to 10.

#### 5. Model Evaluation:

- After training, the model is evaluated on the test data to determine its accuracy.

#### 6. Model Saving:

- The trained model is saved as a `.h5` file for future use or deployment.

## CSS Provided

This styling provides a clean, modern look to the layout with a focus on readability and interactivity.

## HTML Structure and Functionality

#### Webpage Structure:

- **Left Sidebar**: Displays melanoma prevention tips.

- **Main Content**: Includes an image upload form, image preview, and prediction result.

- **Right Sidebar**: Shows suggested actions based on the prediction and a disclaimer.


#### File Upload:

- Users upload an image of a skin lesion, and a preview is shown on the page.

#### Backend Prediction:

- The image is sent to the backend using a `fetch()` POST request for prediction.

- The backend returns the prediction (benign or malignant) and confidence level.

#### Result Handling:

- Displays prediction and confidence percentage.

- Suggested actions are updated:
  - **Malignant**: Consult a dermatologist and take preventive actions.
  - **Benign**: Monitor the lesion and schedule regular skin checks.

 ## Answer of The Question

  Our model demonstrates more than 90% meaningful predictive power in the final presentation, and successfully uploaded and displayed the prediction results on the final webpage. Theoretically, using machine learning for skin cancer prediction is feasible, and our model and results strongly support this view. However, practically speaking, whether machine learning alone can serve as a medical basis for diagnosing skin cancer is still uncertain. There is much more to be done in the field of machine learning and data analysis, and this will be our focus for future efforts.

## References for the data

Our original data from:
[Melanoma Detection Data](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)

## Declaration

During this machine learning task, we used various techniques, algorithms, and tools. Some of them were covered in class, while others were learned through web searches and research. Whenever we encountered an error or faced confusion, we consulted ChatGPT for suggestions or assistance with debugging. We made sure to first understand the issue and its underlying concepts before implementing the solution. Given that we are beginners in machine learning, we had to invest a significant amount of time researching and reading to solve problems. As a result, our work may have similarities with others, but it reflects our learning process and gradual understanding of the subject.

