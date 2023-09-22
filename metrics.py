from sklearn.metrics import accuracy_score

def evaluate(y_true:list, y_pred:list):
    assert len(y_true) == len(y_pred), "y_true and y_pred do not have the same length"
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    return metrics