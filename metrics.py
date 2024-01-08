from utils import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import CLASSES
import os
import re


def generate_confussion_matrix(y_true: list, y_predicted: list, filename: str):
    """Generate confussion_matrix matrix and save it to pickle file.

        :param y_predicted: predicted classes by neural network
        :type y_predicted: list 1-D of integers in range (0, 9)
        :param y_predicted: labeled classes in test set
        :type y_predicted: list 1-D of integers in range (0, 9)
    """
    cf_matrix = confusion_matrix(y_true, y_predicted)

    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in CLASSES],
        columns=[i for i in CLASSES]
    )
    df_cm.to_pickle(f"{filename}-matrix.pickle")





def __count_precision(conf_matrix: np.ndarray) -> list:

    precision = []
    for i in range(len(conf_matrix)):
        if sum(conf_matrix[:, i]) == 0:
            precision.append(0)
        else:
            precision.append(conf_matrix[i, i] / sum(conf_matrix[:, i]))
    return precision


def __count_recall(conf_matrix: np.ndarray) -> list:
    recall = []
    for i in range(len(conf_matrix)):
        if sum(conf_matrix[i, :]) == 0:
            recall.append(0)
        else:
            recall.append(conf_matrix[i, i] / sum(conf_matrix[i, :]))
    return recall


def generate_statistics(options: dict) -> None:
    """Generate statistics and save them as png file.

        :param options: options informing us about where are
        matrixes, models saved and which strategy is being plot.
        :type options: dict
    """
    files = os.listdir(options.get("model_dir"))
    matrixes = os.listdir(options.get("source"))
    filtered_models = sorted([file for file in files if options.get("strategy") in file])
    filtered_matrixes = sorted([matrix for matrix in matrixes if options.get("strategy") in matrix])
    print(filtered_models)    
    # return 
    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f"Strategy: {options.get('strategy')}", fontsize=16)
    
    titles = ["No action", "Show often imbalanced class", "SMOTE data generated", "Customized loss for imbalanced class" ]
    for i in range(4):
        print(filtered_matrixes[i])
        df = pd.read_pickle(f"{options.get('source')}/{filtered_matrixes[i]}")
        _, epoch, acc, loss = load_model(f"{options.get('model_dir')}/{filtered_models[i]}", "cuda:0")
        axs[i, 0].set_title("strategy: {}, accuracy: {} epoch: {} loss: {}".format(titles[i], acc, epoch, loss))
        sn.heatmap(
            df, annot=True, ax=axs[i, 0]
        )
        conf_matrix = df.values
        bar_width = 0.35
        index = list(range(len(conf_matrix)))

        precision = __count_precision(conf_matrix)
        recall = __count_recall(conf_matrix)

        axs[i, 1].bar(index, precision, bar_width, label='Precision', color='skyblue')
        axs[i, 1].bar([i + bar_width for i in index], recall, bar_width, label='Recall', color='lightgreen')
        axs[i, 1].legend()
        axs[i, 1].grid()
        axs[i, 1].set_xlabel("classes")
        axs[i, 1].set_ylabel("Precision/Recall values")
        axs[i, 1].set_xticks(range(len(conf_matrix)), CLASSES)

    os.makedirs(options.get("save_dir"), exist_ok=True)
    plt.savefig(f"{options.get('save_dir')}/{options.get('strategy')}-statistics.png")