import utils
from utils import *


#__________________________________Recurrence___________________________________

Recurrence_data = Load_data('Datasets/Merged_Rec.csv')

# Prepare the data for each model features and target
X, y = prepare_Rec_data(Recurrence_data)

# Columns to one-hot encode
columns_to_encode = ['Menopausal Status','Tumor Size','Lymph Node Status', 'Tumor Grade', 'Histological Type',
                     'ER', 'PR','HER2','Tumor Location','Overall Survival Status']

X = one_hot_encode_columns(X, columns_to_encode)

# Target encoding
y = label_encode_target(y)

# Train test split
X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train[['Overall Survival Status (Month)', 'Recurrence Free Status (Month)']])
X_train_scaled = scaler.transform(X_train[['Overall Survival Status (Month)', 'Recurrence Free Status (Month)']])
X_test_scaled = scaler.transform(X_test[['Overall Survival Status (Month)', 'Recurrence Free Status (Month)']])


def train_Rec_xgboost_model(X_train, y_train, X_test, y_test):
    # Initialize and train XGBoost model
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1)
    utils.train_model_save(xgb_model, X_train, y_train, 'model_Rec_xgb.pkl')
    
    # Training and Testing accuracy
    utils.evaluate_accuracy(xgb_model, X_train, y_train, X_test, y_test, 'model_Rec_xgb')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(xgb_model, X_test, y_test, 'model_Rec_xgb')


def train_Rec_random_forest_model(X_train, y_train, X_test, y_test):
    # Initialize and train Random Forest model
    clf = RandomForestClassifier(n_estimators=10000, random_state=42, n_jobs=1)
    utils.train_model_save(clf, X_train, y_train, 'model_Rec_rf.pkl')
    
    # Training and Testing accuracy
    utils.evaluate_accuracy(clf, X_train, y_train, X_test, y_test, 'model_Rec_rf')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(clf, X_test, y_test, 'model_Rec_rf')


def train_Rec_svc_model(X_train, y_train, X_test, y_test):
    # Initialize and train SVC model
    SVM = SVC(kernel='rbf', C=10000, gamma=100, probability=True)
    utils.train_model_save(SVM, X_train, y_train, 'model_Rec_SV.pkl')
    
    # Training and Testing accuracy
    utils.evaluate_accuracy(SVM, X_train, y_train, X_test, y_test, 'model_Rec_SV')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(SVM, X_test, y_test, 'model_Rec_SV')


def train_Rec_NN_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=64):
    # Initialize the model
    model = Sequential()

    # Add layers to the model
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.002), metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Save the model
    utils.train_model_save(model, X_train, y_train, 'model_Rec_NN.pkl')

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predict probabilities and convert to binary predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Generate classification report
    test_report = classification_report(y_test, y_pred)
    
    # Write test metrics and training report to a text file
    if not os.path.exists('Train_Test_Results'):
        os.makedirs('Train_Test_Results')
    
    test_results_path = os.path.join('Train_Test_Results', 'model_Rec_NN_results.txt')

    with open(test_results_path, 'w') as file:
        # Write training report
        file.write("Training Report:\n")
        file.write("Epochs: {}\n".format(epochs))
        file.write("Batch Size: {}\n".format(batch_size))
        file.write("Training Loss: {}\n".format(history.history['loss'][-1]))
        file.write("Training Accuracy: {}\n".format(history.history['accuracy'][-1]))
        file.write("\n")
        
        # Write test results
        file.write(f"Test Loss: {test_loss:.4f}\n")
        file.write("Test Set Accuracy: {}\n".format(test_accuracy))
        file.write("Test Set Classification Report:\n")
        file.write(test_report)

    # Plot confusion matrix and ROC curve
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(np.arange(2), ['Negative', 'Positive'])
    plt.yticks(np.arange(2), ['Negative', 'Positive'])
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black")
    plt.tight_layout()
    plt.savefig(os.path.join('Figures', 'model_Rec_NN_confusion_matrix.png'))
    plt.close()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('Figures', 'model_Rec_NN_roc_curve.png'))
    plt.close()
    
    
#__________________________________Metastasis___________________________________

# Load train
seer_data = utils.Load_data('Datasets/seer_data_filtered.csv')

# Balancing SEER
Mets_balanced_seer = utils.balance_dataset(seer_data, 'AJCC M', 1 , 9000)
bone_balanced_seer = utils.balance_dataset(seer_data, 'Bone_metastasis', 'Yes', 8000)
lung_balanced_seer = utils.balance_dataset(seer_data, 'Lung_metastasis', 'Yes', 5000)
liver_balanced_seer = utils.balance_dataset(seer_data, 'Liver_metastasis', 'Yes', 5000)
brain_balanced_seer = utils.balance_dataset(seer_data, 'Brain_metastasis', 'Yes', 1000)

# Prepare the data for each model features and target
X_Mets, y_Mets = utils.prepare_Mets_data(Mets_balanced_seer, 'AJCC M')
X_bone, y_bone = utils.prepare_Mets_data(bone_balanced_seer, 'Bone_metastasis')
X_lung, y_lung = utils.prepare_Mets_data(lung_balanced_seer, 'Lung_metastasis')
X_liver, y_liver = utils.prepare_Mets_data(liver_balanced_seer, 'Liver_metastasis')
X_brain, y_brain = utils.prepare_Mets_data(brain_balanced_seer, 'Brain_metastasis')


# Columns to one-hot encode
columns_to_encode = ['Age','Laterality','Grade', 'AJCC_T_stage','AJCC_N_stage', 'Surgery',
                     'Chemotherapy','Radiotherapy', 'Breast_subtype','ER','PR','HER2','Histologic_Type']

X_Mets = utils.one_hot_encode_columns(X_Mets, columns_to_encode)
X_bone = utils.one_hot_encode_columns(X_bone, columns_to_encode)
X_lung = utils.one_hot_encode_columns(X_lung, columns_to_encode)
X_liver = utils.one_hot_encode_columns(X_liver, columns_to_encode)
X_brain = utils.one_hot_encode_columns(X_brain, columns_to_encode)

# Target encoding
y_Mets = utils.label_encode_target(y_Mets)
y_bone = utils.label_encode_target(y_bone)
y_lung = utils.label_encode_target(y_lung)
y_liver = utils.label_encode_target(y_liver)
y_brain = utils.label_encode_target(y_brain)


# Train test split
X_Mets_train, X_Mets_test, y_Mets_train, y_Mets_test = utils.perform_train_test_split(X_Mets, y_Mets)
X_bone_train, X_bone_test, y_bone_train, y_bone_test = utils.perform_train_test_split(X_bone, y_bone)
X_lung_train, X_lung_test, y_lung_train, y_lung_test = utils.perform_train_test_split(X_lung, y_lung)
X_liver_train, X_liver_test, y_liver_train, y_liver_test =  utils.perform_train_test_split(X_liver, y_liver)
X_brain_train, X_brain_test, y_brain_train, y_brain_test =  utils.perform_train_test_split(X_brain, y_brain)





def train_Mets_xgboost_model(X_train, y_train, X_test, y_test, organ_name):
    # Train XGBoost model
    model = xgb.XGBClassifier()
    utils.train_model_save(model, X_train, y_train, f'model_{organ_name}_XG.pkl')
    
    # Training and Testing accuracy
    utils.evaluate_accuracy(model, X_train, y_train, X_test, y_test, f'{organ_name}')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(model, X_test, y_test, f'{organ_name}_XG')



def train_Mets_random_forest_model(X_train, y_train, X_test, y_test, organ_name):
    # Train Random Forest model
    model = RandomForestClassifier()
    utils.train_model_save(model, X_train, y_train, f'model_{organ_name}_rf.pkl')
    
    # Training and Testing accuracy
    utils.evaluate_accuracy(model, X_train, y_train, X_test, y_test, f'{organ_name}')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(model, X_test, y_test, f'{organ_name}_rf')



def train_Mets_svc_model(X_train, y_train, X_test, y_test, organ_name):
    # Train SVC model
    model = SVC(probability=True)
    utils.train_model_save(model, X_train, y_train, f'model_{organ_name}_SVC.pkl')
    
    # Training and Testing accuracy
    utils.evaluate_accuracy(model, X_train, y_train, X_test, y_test, f'{organ_name}')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(model, X_test, y_test, f'{organ_name}_SVC')
