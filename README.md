# Credit Card Fraud Detection 
## Project Overview
This project aims to develop a fraud detection model using various machine learning algorithms to predict whether a credit card transaction is fraudulent or not. The dataset used is the Credit Card Fraud Detection dataset from Kaggle, which contains transactions made by credit cards, and the goal is to identify fraudulent transactions.
In this project, we use several machine learning models, including:
- Logistic Regression
- Neural Networks (Shallow Neural Network)
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)

Additionally, data preprocessing techniques like feature scaling and splitting the dataset into training, testing, and validation sets are applied to improve the model's performance.

## Steps Taken
1. Setup Kaggle API Credentials
The Kaggle dataset is accessed using the Kaggle API. To begin, we created a directory for the Kaggle API credentials and set appropriate permissions.

```
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

2. Download and Unzip Dataset
The Credit Card Fraud Detection dataset is downloaded using the Kaggle API.
```
!kaggle datasets download -d mlg-ulb/creditcardfraud
!unzip creditcardfraud.zip
```

3. Data Preprocessing
The dataset was preprocessed in the following steps:
- RobustScaler was applied to the 'Amount' column to scale it and reduce the impact of outliers.
- Min-Max normalization was applied to the 'Time' column to ensure all features are on a similar scale.
- The data was shuffled to remove any inherent ordering.
- The dataset was split into train, test, and validation sets to evaluate model performance.

```
# Scaling and normalization
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1, 1))
new_df['Time'] = (new_df['Time'] - new_df['Time'].min()) / (new_df['Time'].max() - new_df['Time'].min())

# Shuffling and splitting
new_df = new_df.sample(frac=1, random_state=1)
train, test, val = new_df[:240000], new_df[240000:262000], new_df[262000:]

# Train-test-validation split into X (features) and y (target)
x_train, y_train = train_np[:, :-1], train_np[:, -1]
x_test, y_test = test_np[:, :-1], test_np[:, -1]
x_val, y_val = val_np[:, :-1], val_np[:, -1]
```

4. Model Training and Evaluation
Several models were trained and evaluated based on their performance on the validation set.

Logistic Regression
A Logistic Regression model was trained, and its performance was evaluated using precision, recall, and F1-score. The model performed excellently for non-fraudulent transactions but struggled with fraud detection.
```
# Logistic Regression results
print(classification_report(y_val, logistic_model.predict(x_val), target_names=['Not Fraud', 'Fraud']))
```

Logistic Regression Results:
- Precision (Not Fraud): 1.00
- Precision (Fraud): 0.83
- Recall (Fraud): 0.56
- F1-Score (Fraud): 0.67

Shallow Neural Network
A shallow neural network with one hidden layer was created and trained. The model showed improved recall and precision for fraud detection compared to Logistic Regression.
```
# Neural Network results
print(classification_report(y_val, neural_net_predictions(shallow_nn, x_val), target_names=['Not Fraud', 'Fraud']))
```

Shallow Neural Network Results:
- Precision (Not Fraud): 1.00
- Precision (Fraud): 0.70
- Recall (Fraud): 0.78
- F1-Score (Fraud): 0.74

Random Forest Classifier
The Random Forest Classifier model was trained and evaluated. It showed high performance for non-fraudulent transactions but struggled to detect fraud accurately.
```
# Random Forest results
print(classification_report(y_val, rf.predict(x_val), target_names=['Not Fraud', 'Fraud']))
```

Random Forest Results:
- Precision (Not Fraud): 1.00
- Precision (Fraud): 0.77
- Recall (Fraud): 0.47
- F1-Score (Fraud): 0.59

Gradient Boosting Classifier
The Gradient Boosting Classifier achieved a reasonable balance between precision and recall, but still, the fraud detection performance was suboptimal.
```
# Gradient Boosting results
print(classification_report(y_val, gbc.predict(x_val), target_names=['Not Fraud', 'Fraud']))
```

Gradient Boosting Results:
- Precision (Not Fraud): 1.00
- Precision (Fraud): 0.67
- Recall (Fraud): 0.67
- F1-Score (Fraud): 0.67

Support Vector Classifier (SVC)
The SVC with a balanced class weight performed well for non-fraudulent transactions but showed a low F1 score for fraud detection.
```
# SVC results
print(classification_report(y_val, svc.predict(x_val), target_names=['Not Fraud', 'Fraud']))
```

SVC Results:
- Precision (Not Fraud): 1.00
- Precision (Fraud): 0.07
- Recall (Fraud): 0.97
- F1-Score (Fraud): 0.14

5. Model Comparison
The models were compared based on their ability to detect fraudulent transactions, particularly focusing on the Recall and F1-score for fraud. A high recall is essential for fraud detection, as it ensures that as many fraudulent transactions as possible are identified.
- Logistic Regression showed high precision for non-fraud but struggled with fraud detection (low recall).
- Shallow Neural Network improved the recall for fraud but had slightly lower precision compared to Logistic Regression.
- Random Forest and Gradient Boosting provided a reasonable balance, but both struggled with detecting fraud as well.
- SVC had the highest recall for fraud, but its low precision resulted in poor overall performance.

## Conclusion
In conclusion, while all models performed well in terms of precision for non-fraudulent transactions, none of the models performed exceptionally well at detecting fraudulent transactions, particularly due to the highly imbalanced dataset. The Shallow Neural Network model achieved the best balance between precision and recall for fraud detection.

# Credit Card Fraud Detection with Data Balancing

Project Overview
This project aims to develop a fraud detection model for credit card transactions using various machine learning algorithms. The focus of this part of the project is on handling the class imbalance issue in the dataset, where fraudulent transactions are rare compared to non-fraudulent transactions. The dataset is first balanced using undersampling to ensure that the models are trained on a more balanced dataset, which improves the performance of fraud detection models.

The following algorithms are used:
- Logistic Regression
- Shallow Neural Network (with and without Batch Normalization)
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Classifier (SVC)

## Data Balancing
Before training the models, the data was balanced by undersampling the Not Fraud class to match the number of Fraud instances. Here's the process:

1. Balancing the Dataset
First, we split the dataset into two parts:
- not_frauds: All non-fraudulent transactions.
- frauds: All fraudulent transactions.
```
not_frauds = new_df.query('Class == 0')
frauds = new_df.query('Class == 1')

# Balance the dataset by undersampling the majority class (Not Fraud)
balanced_df = pd.concat([frauds, not_frauds.sample(len(frauds), random_state=1)])
balanced_df['Class'].value_counts()
```
The dataset is then shuffled to ensure that the order of the transactions does not affect the model's performance.
```
balanced_df = balanced_df.sample(frac=1, random_state=1)
```

2. Splitting the Dataset
After balancing, the dataset is split into training, testing, and validation sets.
```
balanced_df_np = balanced_df.to_numpy()

x_train_b, y_train_b = balanced_df_np[:700, :-1], balanced_df_np[:700, -1].astype(int)
x_test_b, y_test_b = balanced_df_np[700:842, :-1], balanced_df_np[700:842, -1].astype(int)
x_val_b, y_val_b = balanced_df_np[842:, :-1], balanced_df_np[842:, -1].astype(int)
```

3. Model Training and Evaluation
The models were trained on the balanced dataset, and their performances were evaluated using precision, recall, and F1-score.
Logistic Regression (Balanced)
```
logistic_model_b = LogisticRegression()
logistic_model_b.fit(x_train_b, y_train_b)

# Evaluate performance on the validation set
print(classification_report(y_val_b, logistic_model_b.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
```

Logistic Regression Results (Balanced Data):
- Precision (Not Fraud): 0.96
- Precision (Fraud): 0.93
- Recall (Fraud): 0.96
- F1-Score (Fraud): 0.94

Shallow Neural Network (Balanced)
```
# Define the shallow neural network model
shallow_nn_b = Sequential()
shallow_nn_b.add(InputLayer(input_shape=(x_train_b.shape[1],)))  # Input layer
shallow_nn_b.add(Dense(2, activation='relu'))  # Hidden layer with ReLU
shallow_nn_b.add(BatchNormalization())  # Batch Normalization
shallow_nn_b.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile and train the model
shallow_nn_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
shallow_nn_b.fit(x_train_b, y_train_b, validation_data=(x_val_b, y_val_b), epochs=40, callbacks=[checkpoint])

# Evaluate performance on the validation set
print(classification_report(y_val_b, neural_net_predictions(shallow_nn_b, x_val_b), target_names=['Not Fraud', 'Fraud']))
```

Shallow Neural Network Results (Balanced Data):
- Precision (Not Fraud): 0.96
- Precision (Fraud): 0.92
- Recall (Fraud): 0.96
- F1-Score (Fraud): 0.94

Random Forest Classifier (Balanced)
```
rf_b = RandomForestClassifier(max_depth=2, n_jobs=-1)
rf_b.fit(x_train_b, y_train_b)

# Evaluate performance on the validation set
print(classification_report(y_val_b, rf_b.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
```

Random Forest Classifier Results (Balanced Data):
- Precision (Not Fraud): 0.70
- Precision (Fraud): 1.00
- Recall (Fraud): 0.56
- F1-Score (Fraud): 0.72

Gradient Boosting Classifier (Balanced)
```
gbc_b = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0)
gbc_b.fit(x_train_b, y_train_b)

# Evaluate performance on the validation set
print(classification_report(y_val_b, gbc_b.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
```

Gradient Boosting Classifier Results (Balanced Data):
- Precision (Not Fraud): 0.81
- Precision (Fraud): 1.00
- Recall (Fraud): 0.76
- F1-Score (Fraud): 0.86

Support Vector Classifier (SVC) (Balanced)
```
svc_b = LinearSVC(class_weight='balanced')
svc_b.fit(x_train_b, y_train_b)

# Evaluate performance on the validation set
print(classification_report(y_val_b, svc_b.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
```

Support Vector Classifier (SVC) Results (Balanced Data):
- Precision (Not Fraud): 0.95
- Precision (Fraud): 0.97
- Recall (Fraud): 0.94
- F1-Score (Fraud): 0.96

4. Performance on the Test Set
I also evaluated the models on the test set to see how they generalized to unseen data.
```
print(classification_report(y_test_b, neural_net_predictions(shallow_nn_b, x_test_b), target_names=['Not Fraud', 'Fraud']))
```

Shallow Neural Network (Test Set):
- Precision (Not Fraud): 0.92
- Precision (Fraud): 0.94
- Recall (Fraud): 0.91
- F1-Score (Fraud): 0.93

## Conclusion
Balancing the dataset significantly improved the performance of the models, particularly in detecting fraudulent transactions. The Shallow Neural Network and Support Vector Classifier (SVC) showed excellent performance with high precision and recall for fraudulent transactions.

## Final Model Recommendation
- Shallow Neural Network: Achieved a good balance between precision, recall, and F1-score, making it suitable for deployment.
- Support Vector Classifier (SVC): Showed the highest F1-score for fraud detection, making it a strong candidate for further optimization.

## Future Work:
- Further explore advanced techniques like SMOTE for synthetic oversampling of the minority class.
- Experiment with more complex models like deep learning (e.g., CNNs, RNNs) for improved fraud detection.
- Implement hyperparameter optimization to further fine-tune the models and improve their performance.
