from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def compute_metrics(y_true, y_pred) -> dict:
    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC-Score": roc_auc_score(y_true, y_pred),
        "CM true positive": conf_matrix[0][0],
        "CM true negative": conf_matrix[1][1],
        "CM false positive": conf_matrix[0][1],
        "CM false negative": conf_matrix[1][0],
    }

    return metrics
