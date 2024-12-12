# ML-Melanoma-Detection-Model
## Table of Contents
- [Introduction](#Introduction)
- [Tools](#Tools)
- [Model Quick Start Guide](#model-quick-start-guide)
  - [Step 1: Access the Dataset](#step-1-access-the-dataset)
  - [Step 2: Download the model_file](#step-2-download-the-model_file)
  - [Step 3: Launch a Kaggle Notebook](#step-3-launch-a-kaggle-notebook)
- [Key Steps](#Key-Steps)
  - [Model Creation Process](#Model-Creation-Process)
    - [1. Data Preparation](#1-Data-Preparation)
    - [2. Model Architecture](#2-Model-Architecture)
    - [3. Model Compilation](#3-Model-Compilation)
    - [4. Model Training](#4-Model-Training)
    - [5. Model Evaluation](#5-Model-Evaluation)
    - [6. Model Saving](#6-Model-Saving)
   - [CSS Provided](#CSS-Provided)
  - [HTML Structure and Functionality](#HTML-Structure-and-Functionality)
  - [Webpage Structure](#Webpage-Structure)
  - [File Upload](#File-Upload)
  - [Backend Prediction](#Backend-Prediction)
  - [Result Handling](#Result-Handling)
- [Answer of The Question](#Answer-of-The-Question)
- [References for the Data](#References-for-the-Data)
- [Declaration](#Declaration)

## Introduction
Melanoma is a highly aggressive type of skin cancer characterized by rapid growth and early potential to spread to lymph nodes and distant organs. The primary risk factor is excessive exposure to ultraviolet (UV) radiation, particularly from intermittent intense sun exposure or frequent use of tanning beds. Patients typically present with skin lesions that are asymmetrical, have irregular borders, show color variations, or are larger than 6 millimeters in diameter. Additionally, existing moles that exhibit rapid changes can also be indicative of melanoma. The early symptoms of melanoma are often difficult to detect with the naked eye, leading many patients to seek medical attention only in the late stages of the disease, missing the optimal treatment window. Traditional diagnosis relies on experienced dermatologists, but their assessments can be limited by the following factors:   
Subjectivity: Diagnostic results can vary between doctors due to differences in experience and perspective.   
Resource Scarcity: In many remote or under-resourced areas, the number of professional dermatologists is insufficient, resulting in delayed diagnoses.   
Meanwhile, machine learning has the potential to efficiently and accurately process large-scale data. Could it be used to improve the speed and efficiency of screening, thereby helping people with early prevention, diagnosis, and reducing unnecessary medical costs and burdens?

This project aims to answer the question: **Can we use machine learning to build a model that determines the likelihood of melanoma by uploading images of the skin?** This would help people detect skin cancer early and address the shortcomings of traditional medical resources.

## Tools
kaggle，google colab

## Model Quick Start Guide
### Step 1: Access the Dataset
1. Visit the dataset's Kaggle page:  
   [Melanoma Skin Cancer Dataset of 10,000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)

2. **Sign in to Kaggle** if you haven’t already.  
   - If you don’t have an account, create one at [Kaggle](https://www.kaggle.com).

### Step 2: Download the `model_file`
1. Locate the `model_file` folder within the dataset page.  
   - **Download** the file(s) contained in this folder to access the code for the model.

### Step 3: Launch a Kaggle Notebook
1. On the same Kaggle page, click the **"New Notebook"** button.  
   - This will open a new Jupyter Notebook hosted on Kaggle.

2. In the new notebook, **copy and paste the code** from the `model_file` folder into a cell.

3. **Run the Notebook** to execute the model code.  
   - Kaggle’s integrated environment will handle the dependencies and access the dataset without requiring local downloads of the entire dataset.

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

### CSS Provided

This styling provides a clean, modern look to the layout with a focus on readability and interactivity.

### HTML Structure and Functionality

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

Our model demonstrates more than 90% meaningful predictive power in the final presentation, and successfully uploaded and displayed the prediction results on the final webpage. Theoretically, using machine learning for skin cancer prediction is feasible, and our model and results strongly support this view. However, from a practical perspective, whether machine learning alone can serve as a medical basis for diagnosing melanoma still requires more data development and practical validation. There is much more to be done in the field of machine learning and data analysis, and this will be our focus for future efforts.Nevertheless, the journey ahead is long and challenging, but we will persevere in our pursuit. Being able to help more people through such a convenient method is the original intention of many data analysts when they embark on their learning journey. It is also the purpose behind our choice of this topic.

## References for the data

Our original data from:
[Melanoma Detection Data](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)

## Declaration

During this machine learning task, we used various techniques, algorithms, and tools. Some of them were covered in class, while others were learned through web searches and research. Whenever we encountered an error or faced confusion, we consulted ChatGPT for suggestions or assistance with debugging. We made sure to first understand the issue and its underlying concepts before implementing the solution. Given that we are beginners in machine learning, we had to invest a significant amount of time researching and reading to solve problems. As a result, our work may have similarities with others, but it reflects our learning process and gradual understanding of the subject.

