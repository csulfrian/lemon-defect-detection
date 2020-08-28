import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, r2_score, roc_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.decomposition import PCA

import sys
sys.path.append('./src')
# from img_proc import ImagePreprocessor
from dataset_prep import AnnotationsParser, DatasetBuilder
# from eda import MakePlots


def get_scores(y_true, y_pred, average='micro'):
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)

    conf = confusion_matrix(y_true, y_pred)

    return recall, precision, accuracy, f1, conf

def make_y():
    y_ = pd.read_csv('data/processed/y.csv').set_index('Unnamed: 0')
    y = y_['0'].to_numpy()

    return y


if __name__ == '__main__':
    ann_dir = 'data/raw/annotations'
    fname = 'instances_default.json'

    lemons = DatasetBuilder(ann_dir, fname)
    X = lemons.load_data()
    y = make_y()

    # print('\nDoing PCA...')
    # pca = PCA(n_components=0.9)  
    # X_pca = pca.fit_transform(X)
    # print(f'Number of components covering 90% variance: {pca.n_components_}')
    # print(f'Variance components: {np.around(pca.explained_variance_[0:6], 1)}')

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


    l1s = [0.01, 0.1, 0.5, 0.9, 0.99]
    # pipeline = Pipeline()
    clf = LogisticRegressionCV(max_iter=1000,
                               tol=0.001,
                               n_jobs=-1,
                               penalty='elasticnet',
                               l1_ratios=l1s,
                               multi_class='multinomial',
                               solver='saga',
                               verbose=1)
    print('Making CV\n')
    C = clf.fit(X_train, y_train)

    # regr = LogisticRegression(class_weight='balanced',
    #                           max_iter=2000,
    #                           penalty='elasticnet',
    #                           l1_ratio=0.1,
    #                           solver='saga',
    #                           multi_class='multinomial',
    #                           n_jobs=-1)

    # print('\nFitting model...\n')
    # regr.fit(X_train, y_train)

    print('\nPredicting!')
    probs = clf.predict_proba(X_test)
    pred_classes = clf.predict(X_test)

    print(f'Probabilities: {probs}')
    print(f'Predicted classes: {pred_classes}')

    avg_type = 'micro'
    recall, precision, accuracy, f1, conf= get_scores(y_test, pred_classes, avg_type)

    print(f'Recall: {round(recall, 2)}')
    print(f'Precision: {round(precision, 2)}')
    print(f'Accuracy: {accuracy}')
    print(f'F1 score: {f1}')
    print(f'Confusion matrix: {conf}')