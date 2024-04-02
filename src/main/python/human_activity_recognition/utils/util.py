# --------------------------------------------------------------------------------
# BodyFlow
# Version: 2.0
# Copyright (c) 2024 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: March 2024
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@ita.es
#          Angel Gimeno Valero                              agimeno@ita.es
#          Carlos Maranes Nueno                            cmaranes@ita.es
#          Irene Lopez Bosque                                ilopez@ita.es
#          Jose Ignacio Calvo Callejo                       jicalvo@ita.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@ita.es
#          Pilar Salvo Ibanez                                psalvo@ita.es
#          Rafael del Hoyo Alonso                          rdelhoyo@ita.es
#          Rocio Aznar Gimeno                                raznar@ita.es
#          Pablo Perez Lazaro                               plazaro@ita.es
#          Marcos Marina Castello                           mmarina@ita.es
# All rights reserved 
# --------------------------------------------------------------------------------

import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import logging
from mlflow import MlflowClient
import os
import mlflow
import logging



def plot_confusion_matrix(y_true, y_pred, classes, har_model, input_data, window_size):
    cf_matrix_norm = confusion_matrix(y_true, y_pred, normalize = 'true')
    cf_matrix = confusion_matrix(y_true, y_pred)
    logging.info('Confusion matrix')
    logging.info(cf_matrix_norm)
    
    fig = plt.figure(figsize = (8, 8))
    ax = sns.heatmap(cf_matrix_norm, fmt = '.2f', annot = True, annot_kws = {'fontsize': 8},
                     xticklabels = classes, yticklabels = classes)
    ax.set(xlabel = 'True Label', ylabel = 'Predicted Label')
    matrix_title =  f'HAR Model {har_model} \nInput Data {input_data}'
    ax.set(title = matrix_title)

    plt.xticks(fontsize = 8) 
    plt.yticks(fontsize = 8) 
    temp_name = "confusion-matrix.png"
    plt.savefig(temp_name)
    plt.close(fig)

    # Log the confusion matrix to mlflow
    mlflow.log_artifact(temp_name, 'confusion-matrix-plots')
    try:
        os.remove(temp_name)
    except FileNotFoundError:
        logging.warning(f"{temp_name} file is not found")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    txt_file =  f'har_{har_model}_{input_data}_{window_size}.txt'
    accuracy = accuracy_score(y_true, y_pred)
    
    with open(txt_file, 'a') as file:
        file.write(f'{har_model} {input_data} {window_size}\n')
        file.write(f'Confusion Matrix: \n{cf_matrix}\n') 
        file.write(f'Confusion Matrix Normalized: \n{cf_matrix_norm}\n') 
        file.write(f'Accuracy: {accuracy}\n')
    
    for type_eror in ['macro', 'weighted']:
        precision = precision_score(y_true, y_pred, average=type_eror)  # 'weighted' for multiclass
        recall = recall_score(y_true, y_pred, average=type_eror)  # 'weighted' for multiclass
        f1 = f1_score(y_true, y_pred, average=type_eror)  # 'weighted' for multiclass

        with open(txt_file, 'a') as file:
            file.write(f'Precision {type_eror}: {precision}\n')
            file.write(f'Recall {type_eror}: {recall}\n')
            file.write(f'F1 Score {type_eror}: {f1}\n')

    mlflow.log_artifact(txt_file, 'final_results')
    try:
        os.remove(txt_file)
    except FileNotFoundError:
        logging.warning(f"{txt_file} file is not found")
    return



def auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    
    best_metric_key = max(r.data.metrics, key=lambda k: r.data.metrics[k])
    best_metric_value = r.data.metrics[best_metric_key]
    print(best_metric_key, best_metric_value)
    
    logging.info("run_id: {}".format(r.info.run_id))
    logging.info("artifacts: {}".format(artifacts))
    logging.info("params: {}".format(r.data.params))
    logging.info("metrics: {}".format(r.data.metrics))
    logging.info("tags: {}".format(tags))

    return


def predictions_to_csv(df_test, y_pred):
    # Generate csv with the activity predictions added to the test dataset
    df_test['activity_prediction'] = y_pred
    df_test['activity_prediction'] += 1

    temp_name = 'output_activity_predictions.csv'
    df_test.to_csv(temp_name, index=False)
    # Log the confusion matrix to mlflow
    mlflow.log_artifact(temp_name, 'test_predictions')
    try:
        os.remove(temp_name)
    except FileNotFoundError:
        logging.warning(f"{temp_name} file is not found")
    return




def save_confusion_matrix(y_true, y_pred, classes, har_model, input_data, window_size, saving_path):
    cf_matrix_norm = confusion_matrix(y_true, y_pred, normalize = 'true')
    cf_matrix = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize = (8, 8))
    ax = sns.heatmap(cf_matrix_norm, fmt = '.2f', annot = True, annot_kws = {'fontsize': 8},
                     xticklabels = classes, yticklabels = classes)
    ax.set(xlabel = 'True Label', ylabel = 'Predicted Label')
    matrix_title =  f'HAR Model {har_model} \nInput Data {input_data}'
    ax.set(title = matrix_title)

    plt.xticks(fontsize = 8) 
    plt.yticks(fontsize = 8) 
    temp_name = f"confusion-matrix_{har_model}_{input_data}_{window_size}.png"
    plt.savefig(os.path.join(saving_path, temp_name)) 
    plt.close(fig)
    
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    txt_file = os.path.join(saving_path, f'har_{har_model}_{input_data}_{window_size}.txt')
    accuracy = accuracy_score(y_true, y_pred)
    
    with open(txt_file, 'a') as file:
        file.write(f'{har_model} {input_data} {window_size}\n')
        file.write(f'Confusion Matrix: \n{cf_matrix}\n') 
        file.write(f'Confusion Matrix Normalized: \n{cf_matrix_norm}\n') 
        file.write(f'Accuracy: {accuracy}\n')
    
    for type_eror in ['macro', 'weighted']:
        precision = precision_score(y_true, y_pred, average=type_eror)  # 'weighted' for multiclass
        recall = recall_score(y_true, y_pred, average=type_eror)  # 'weighted' for multiclass
        f1 = f1_score(y_true, y_pred, average=type_eror)  # 'weighted' for multiclass

        with open(txt_file, 'a') as file:
            file.write(f'Precision {type_eror}: {precision}\n')
            file.write(f'Recall {type_eror}: {recall}\n')
            file.write(f'F1 Score {type_eror}: {f1}\n')

    return