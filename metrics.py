from sklearn.metrics import accuracy_score

def evaluate(y_true, y_pred):
    results = {}
    results['pred'] = None