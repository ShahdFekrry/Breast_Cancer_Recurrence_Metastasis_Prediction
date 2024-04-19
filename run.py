import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from train_test import *
from Validation import *
def main():
    print("Welcome to the Breast Cancer Prediction Tool!")

    # Step 1: Choose between recurrence or metastasis
    print("Select the prediction task:")
    print("1. Recurrence")
    print("2. Metastasis")
    task_choice = input("Enter your choice (1 or 2): ")
    while task_choice not in ['1', '2']:
        print("Invalid choice. Please enter either '1' or '2'.")
        task_choice = input("Enter your choice (1 or 2): ")

    prediction_task = 'recurrence' if task_choice == '1' else 'metastasis'

    # Step 2: Choose between training or validation
    print("Select the task type:")
    print("1. Train and Testing")
    print("2. Validate")
    task_type_choice = input("Enter your choice (1 or 2): ")
    while task_type_choice not in ['1', '2']:
        print("Invalid choice. Please enter either '1' or '2'.")
        task_type_choice = input("Enter your choice (1 or 2): ")

    task_type = 'train' if task_type_choice == '1' else 'validate'

    # Step 3: Execute the chosen task
    if task_type == 'train':
        # Choose which model(s) to train
        print("Select the model(s) to train:")
        if prediction_task == 'recurrence':
            print("1. XGBoost")
            print("2. Random Forest")
            print("3. Support Vector Machine (SVM)")
            print("4. Neural Network (NN)")
            print("5. All of them")
            model_choice = input("Enter your choice (1, 2, 3, 4, or 5): ")
            if model_choice == '1':
                train_Rec_xgboost_model(X_train_scaled, y_train, X_test_scaled, y_test)
            elif model_choice == '2':
                train_Rec_random_forest_model(X_train_scaled, y_train, X_test_scaled, y_test)
            elif model_choice == '3':
                train_Rec_svc_model(X_train_scaled, y_train, X_test_scaled,y_test)
            elif model_choice == '4':
               train_Rec_NN_model(X_train_scaled, y_train, X_test_scaled, y_test)
            elif model_choice == '5':
                train_Rec_xgboost_model(X_train_scaled, y_train, X_test_scaled, y_test)
                train_Rec_random_forest_model(X_train_scaled, y_train, X_test_scaled, y_test)
                train_Rec_svc_model(X_train_scaled, y_train, X_test_scaled, y_test)
                train_Rec_NN_model(X_train_scaled, y_train, X_test_scaled, y_test)
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        else:  # prediction_task == 'metastasis'
            print("1. XGBoost")
            print("2. Random Forest")
            print("3. Support Vector Machine (SVM)")
            print("4. All of them")
            model_choice = input("Enter your choice (1, 2, 3, or 4): ")
            if model_choice == '1':
                train_Mets_xgboost_model(X_Mets_train, y_Mets_train, X_Mets_test, y_Mets_test, 'Mets')
                train_Mets_xgboost_model(X_bone_train,  y_bone_train, X_bone_test, y_bone_test, 'bone')
                train_Mets_xgboost_model(X_lung_train, y_lung_train, X_lung_test, y_lung_test, 'lung')
                train_Mets_xgboost_model(X_liver_train, y_liver_train, X_liver_test, y_liver_test, 'liver')
                train_Mets_xgboost_model(X_brain_train, y_brain_train, X_brain_test, y_brain_test, 'brain')

            elif model_choice == '2':
                train_Mets_random_forest_model(X_Mets_train, y_Mets_train, X_Mets_test, y_Mets_test, 'Mets')
                train_Mets_random_forest_model(X_bone_train, y_bone_train, X_bone_test, y_bone_test, 'bone')
                train_Mets_random_forest_model(X_lung_train, y_lung_train, X_lung_test, y_lung_test, 'lung')
                train_Mets_random_forest_model(X_liver_train, y_liver_train, X_liver_test, y_liver_test, 'liver')
                train_Mets_random_forest_model(X_brain_train, y_brain_train, X_brain_test, y_brain_test, 'brain')

            elif model_choice == '3':
                train_Mets_svc_model(X_Mets_train, y_Mets_train, X_Mets_test, y_Mets_test, 'Mets')
                train_Mets_svc_model(X_bone_train, y_bone_train, X_bone_test, y_bone_test, 'bone')
                train_Mets_svc_model(X_lung_train, y_lung_train, X_lung_test, y_lung_test, 'lung')
                train_Mets_svc_model(X_liver_train, y_liver_train, X_liver_test, y_liver_test, 'liver')
                train_Mets_svc_model(X_brain_train, y_brain_train, X_brain_test, y_brain_test, 'brain')

            elif model_choice == '4':
                #XG
                train_Mets_xgboost_model(X_Mets_train, y_Mets_train, X_Mets_test, y_Mets_test, 'Mets')
                train_Mets_xgboost_model(X_bone_train, y_bone_train, X_bone_test, y_bone_test, 'bone')
                train_Mets_xgboost_model(X_lung_train, y_lung_train, X_lung_test, y_lung_test, 'lung')
                train_Mets_xgboost_model(X_liver_train, y_liver_train, X_liver_test, y_liver_test, 'liver')
                train_Mets_xgboost_model(X_brain_train, y_brain_train, X_brain_test, y_brain_test, 'brain')
                
                # RF
                train_Mets_random_forest_model(X_Mets_train, y_Mets_train, X_Mets_test, y_Mets_test, 'Mets')
                train_Mets_random_forest_model(X_bone_train, y_bone_train, X_bone_test, y_bone_test, 'bone')
                train_Mets_random_forest_model(X_lung_train, y_lung_train, X_lung_test, y_lung_test, 'lung')
                train_Mets_random_forest_model(X_liver_train, y_liver_train, X_liver_test, y_liver_test, 'liver')
                train_Mets_random_forest_model(X_brain_train, y_brain_train, X_brain_test, y_brain_test, 'brain')

                # SVM
                train_Mets_svc_model(X_Mets_train, y_Mets_train, X_Mets_test, y_Mets_test, 'Mets')
                train_Mets_svc_model(X_bone_train, y_bone_train, X_bone_test, y_bone_test, 'bone')
                train_Mets_svc_model(X_lung_train, y_lung_train, X_lung_test, y_lung_test, 'lung')
                train_Mets_svc_model(X_liver_train, y_liver_train, X_liver_test, y_liver_test, 'liver')
                train_Mets_svc_model(X_brain_train, y_brain_train, X_brain_test, y_brain_test, 'brain')

            else:
                print("Invalid choice. Please enter a number between 1 and 4.")

    else: # validation
        if prediction_task == 'recurrence':
            
            print("1. XGBoost")
            print("2. Random Forest")
            print("3. Support Vector Machine (SVM)")
            print("4. Neural Network (NN)")
            print("5. All of them")
            model_choice = input("Enter your choice (1, 2, 3, 4, or 5): ")
            if model_choice == '1':
                validate_Rec_xgboost_model(X_val_scaled, y_val)
            elif model_choice == '2':
                validate_Rec_rf_model(X_val_scaled, y_val)
            elif model_choice == '3':
                validate_Rec_SV_model(X_val_scaled, y_val)
            elif model_choice == '4':
                validate_Rec_NN_model(X_val_scaled, y_val)
            elif model_choice == '5':
                validate_Rec_xgboost_model(X_val_scaled, y_val)
                validate_Rec_rf_model(X_val_scaled, y_val)
                validate_Rec_SV_model(X_val_scaled, y_val)
                validate_Rec_NN_model(X_val_scaled, y_val)
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")

        else: # Metastasis Validation
            print("1. XGBoost")
            print("2. Random Forest")
            print("3. Support Vector Machine (SVM)")
            print("4. Neural Network (NN)")
            print("5. All of them")
            
        
        

if __name__ == "__main__":
    main()