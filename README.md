## Random Forest Classifier for Wine Quality Prediction

This repository contains Python code for training a Random Forest Classifier model to predict the quality of red wine based on certain features. The Random Forest algorithm is a popular machine learning technique that combines multiple decision trees to make accurate predictions. The code provided demonstrates the step-by-step process of loading the dataset, preprocessing the data, training the model, tuning the hyperparameters, and evaluating the accuracy of the model.

### Dataset

The dataset used in this project is the "winequality-red.csv" file, which contains information about various attributes of red wine samples and their corresponding quality ratings. The dataset is loaded into a Pandas DataFrame using the `read_csv` function.

### Preprocessing

Before training the model, some preprocessing steps are performed on the dataset. First, the target variable, "quality," is binned into three categories: "bad," "decent," and "excellent." This helps to transform the problem into a classification task. The `cut` function from Pandas is used for binning the target variable.

Next, the categorical target variable is converted into numeric form using the LabelEncoder from the scikit-learn library. This ensures that the target variable is represented by numerical values that can be used for training the model.

The dataset is then split into training and testing sets using the `train_test_split` function from scikit-learn. This allows us to evaluate the performance of the model on unseen data.

### Model Training

A Random Forest Classifier is employed to train the model. The classifier is initialized with default values, and the `fit` function is used to train the model on the training data.

### Hyperparameter Tuning

To optimize the performance of the Random Forest Classifier, hyperparameter tuning is performed using Grid Search. Grid Search is a technique that exhaustively searches through a specified parameter grid to find the best combination of hyperparameters for the model. In this case, the grid includes parameters like the number of estimators, maximum depth, minimum samples split, and minimum samples leaf. The GridSearchCV class from scikit-learn is used for this purpose.

The best parameters found by Grid Search are printed to the console.

### Model Evaluation

Finally, the accuracy of the model is evaluated using the best parameters obtained from Grid Search. The tuned Random Forest Classifier is fitted on the training data, and predictions are made on the testing data. The accuracy of the predictions is calculated using the `accuracy_score` function from scikit-learn. The accuracy score represents the percentage of correctly predicted wine quality ratings.


### Conclusion

This code provides a clear example of how to implement a Random Forest Classifier for wine quality prediction. By following the steps outlined in the code, you can train a model, tune its hyperparameters, and evaluate its accuracy. The tuned model demonstrates a slightly improved accuracy compared to the default model, showcasing the effectiveness of hyperparameter tuning. Feel free to modify the code according to your specific requirements or extend it for further analysis or visualization.