from typing import List, Tuple


def get_confusion_matrix(
        y_true: List[int], y_pred: List[int], num_classes: int,
) -> List[List[int]]:
    """
    Generate a confusion matrix in a form of a list of lists.

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values
    :param num_classes: number of supported classes

    :return: confusion matrix
    """
    # Checking the shapes
    if len(y_true) != len(y_pred):
        raise ValueError("Invalid input shapes!")

    # Checking if num classes are positive
    if num_classes <= 0:
        raise ValueError("Invalid num_classes!")

    # Checking if all predictions in range
    for pred_class in y_pred:
        if pred_class < 0 or num_classes <= pred_class:
            raise ValueError("Invalid prediction classes!")

    # Init matrix
    confusion_matrix = []
    for _ in range(num_classes):
        row = [0] * num_classes
        confusion_matrix.append(row)

    # Calculate matrix
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1
    return confusion_matrix


def get_quality_factors(
    y_true: List[int],
    y_pred: List[int],
) -> Tuple[int, int, int, int]:
    """
    Calculate True Negative, False Positive, False Negative and True Positive
    metrics basing on the ground truth and predicted lists.

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: a tuple of TN, FP, FN, TP
    """
    # Only 2 possible values sooo...
    tn, fp, fn, tp = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 1 and pred == 1:
            tp += 1
    return tn, fp, fn, tp

def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the accuracy for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: accuracy score
    """
    tn, fp, fn, tp = get_quality_factors(y_true, y_pred)
    return (tp+tn)/len(y_true)

def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the precision for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: precision score
    """
    tn, fp, fn, tp = get_quality_factors(y_true, y_pred)
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)

def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the recall for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: recall score
    """
    tn, fp, fn, tp = get_quality_factors(y_true, y_pred)
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)

def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the F1-score for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: F1-score
    """
    tn, fp, fn, tp = get_quality_factors(y_true, y_pred)
    if (2*tp+fp+fn) == 0:
        return 0
    else:
        return (2*tp)/(2*tp+fp+fn)