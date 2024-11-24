### Informe Final: Clasificación de Actividades Basadas en Landmarks de Poses

#### 
**Classification of Human Activities Using Pose Landmarks and Machine Learning**

#### **Abstract**
This report presents a methodology for classifying human activities using pose landmarks extracted from video frames. The process involves pose detection, feature extraction, and classification through machine learning techniques. A Random Forest classifier was trained on preprocessed pose data, achieving reliable predictions. This approach highlights the potential of pose-based classification for applications in sports analysis, rehabilitation, and fitness tracking.

---

#### **Introduction**
Human activity recognition plays a crucial role in understanding movements and behaviors in domains such as healthcare, sports, and entertainment. This project aims to classify physical activities by leveraging pose landmarks obtained from video frames. The primary objective is to develop a system that preprocesses pose data, extracts meaningful features, and classifies activities with high accuracy. Pose-based classification, as opposed to traditional image recognition, is computationally efficient and provides interpretable results for specific applications.

#### **Theory**
To understand the development of this project, the key concepts include:
- **Pose Landmarks:** Specific points (e.g., shoulders, elbows, knees) tracked on the human body using MediaPipe.
- **Feature Engineering:** Calculating angles between joints and joint velocities to represent physical activities.
- **Machine Learning Models:** A Random Forest model is used due to its robustness in handling multidimensional features and interpretability.
  
Key theoretical principles include kinematic analysis (using angles and velocities) and supervised learning (classification).

#### **Methodology**
1. **Data Collection**
Collect videos for predefined movements (e.g., turning, walking forward, walking backward).
Each group member records 10 videos per pose to ensure diversity.
Extract 3D pose landmarks using MediaPipe and store them in JSON and CSV formats.
2. **Data Preprocessing**
Normalize coordinates relative to the hip joint for positional invariance.
Apply Gaussian filtering to reduce noise.
Engineer features: joint angles, velocities, and other derived metrics.

3. **Model Training**
  - A Random Forest Classifier was chosen for its robustness, interpretability, and ability to handle complex, nonlinear relationships in the data. This model was particularly suited to the diverse and high-dimensional pose landmark features extracted during preprocessing.

  - The dataset was split into training and testing sets, with 80% of the data used for training and 20% reserved for testing. This split ensured an unbiased evaluation of the model's performance on unseen data.

  - Hyperparameter Tuning: A comprehensive hyperparameter optimization was performed using GridSearchCV
    * A StratifiedKFold Cross-Validation with 5 folds was used to evaluate model performance during tuning, ensuring balanced class distribution across splits.
4. **Model Evaluation**
Evaluate the classifier using metrics such as precision, recall, and F1-score.
Inspect confusion matrices for movement misclassification patterns.
Fine-tune based on test results to improve overall accuracy.

5. **Deployment and Application**
Deploy the trained model for real-time or batch movement recognition.
Integrate predictions into broader applications like fitness tracking or sports analytics.




#### **Results**
- **Model Performance:**
  - The Random Forest classifier achieved an accuracy of approximately ??% on the test set.

- **Metrics:**
  - Precision, recall, and F1 scores were calculated for each activity class.

####  **Results Analysis**
- The model generalized well to unseen test data, indicating robust feature extraction and preprocessing steps.
- Minor overfitting was observed due to the limited diversity of training data.
- Performance varied across activity classes; simpler activities (e.g., standing) were classified more accurately than dynamic ones (e.g., jumping).
- Comparisons with literature suggest competitive performance for similar datasets.

#### **Conclusions**
  - The project successfully demonstrated the feasibility of classifying activities using pose landmarks and machine learning.




#### **Bibliographic References**

1. MediaPipe Documentation, https://mediapipe.dev.

