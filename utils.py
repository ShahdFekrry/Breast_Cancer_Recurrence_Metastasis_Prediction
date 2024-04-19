import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os


def Load_data(filename):
    data = pd.read_csv(filename)
    return data


def balance_dataset(data, target_column, target_value, sample_size):
    # Filter data based on target value
    target_data = data[data[target_column] == target_value].dropna(subset=[target_column])

    # Sample from the remaining data
    remaining_data = data[data[target_column] != target_value].dropna(subset=[target_column])
    random_sample = remaining_data.sample(n=sample_size, random_state=42)

    # Concatenate the sampled data with the target data
    balanced_data = pd.concat([random_sample, target_data], axis=0)

    return balanced_data


# Features and Targets
def prepare_Mets_data(df, target):
    X = df.drop(columns=['AJCC M', 'Brain_metastasis', 'Lung_metastasis', 'Liver_metastasis', 'Bone_metastasis'], axis=1)
    y = df[target]
    return X, y

# Features and Targets
def prepare_Rec_data(df):
    X = df.drop('Recurrence Free Status',axis=1)
    y = df['Recurrence Free Status']
    return X, y

# Encoding the features
def one_hot_encode_columns(features, columns):
    return pd.get_dummies(features, columns=columns)

# Label encoding for y
def label_encode_target(y):
    label_encoder = preprocessing.LabelEncoder()
    return label_encoder.fit_transform(y)

# Train test split
def perform_train_test_split(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


# Training and saving weights
def train_model_save(model, X_train, y_train, save_path):
    # Train the model
    model.fit(X_train, y_train)

    # Create the 'Saved Models' folder if it doesn't exist
    if not os.path.exists('Saved Models'):
        os.makedirs('Saved Models')

    # Save the model weights
    joblib.dump(model, f'Saved Models/{save_path}')


def evaluate_accuracy(model, X_train, y_train, X_test, y_test, filename_prefix):
    # Train accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_report = classification_report(y_train, train_pred)

    # Test accuracy
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_report = classification_report(y_test, test_pred)

    # Create the 'Train_Test_Results' folder if it doesn't exist
    if not os.path.exists('Train_Test_Results'):
        os.makedirs('Train_Test_Results')

    # Write results to a text file
    with open(f'Train_Test_Results/{filename_prefix}_results.txt', "w") as file:
        file.write("Train Accuracy: {}\n".format(train_acc))
        file.write("Train Classification Report:\n")
        file.write(train_report)
        file.write("\nTest Accuracy: {}\n".format(test_acc))
        file.write("Test Classification Report:\n")
        file.write(test_report)

    return train_acc, test_acc

def evaluate_validation_accuracy(model, X_val, y_val, filename_prefix):
    # Test accuracy
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    test_report = classification_report(y_val, val_pred)

    # Create the 'Train_Test_Results' folder if it doesn't exist
    if not os.path.exists('External_Validation_Results'):
        os.makedirs('External_Validation_Results')

    # Write results to a text file
    with open(f'External_Validation_Results/{filename_prefix}_results.txt', "a") as file:
        file.write("\nValidation Accuracy: {}\n".format(val_acc))
        file.write("Validation Classification Report:\n")
        file.write(test_report)

    return val_acc

# Confusion and ROC
def plot_evaluation_metrics(model, X_test, y_test, filename_prefix):
    # Confusion Matrix
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    # Create the 'Figures' folder if it doesn't exist
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    
    plt.savefig(f'Figures/{filename_prefix}_confusion_matrix.png')
    plt.close()

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates of the positive class
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.savefig(f'Figures/{filename_prefix}_roc_curve.png')
    plt.close()
