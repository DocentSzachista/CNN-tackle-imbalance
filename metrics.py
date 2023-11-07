from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import CLASSES


def generate_confussion_matrix(y_true: list, y_predicted: list, filename: str):
    """Generate confussion_matrix heatmap.

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
    plt.figure(figsize=(12, 7))
    sn.heatmap(
        df_cm, annot=True
    )
    plt.savefig(f"{filename}.png")
