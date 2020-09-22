import sys
sys.path.append('./src')
from img_proc import ImagePreprocessor
from dataset_prep import DatasetBuilder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score, recall_score,\
                            precision_score, accuracy_score,\
                            f1_score, multilabel_confusion_matrix,\
                            classification_report
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import numpy as np
import pandas as pd
import joblib
import os


def get_scores(y_true, y_pred, average='micro'):
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    jaccard = jaccard_score(y_true, y_pred, average=average)
    target_names = ['Inedible', 'Commercial', 'Retail']
    report = classification_report(y_test, y_pred,
                                   target_names=target_names,
                                   digits=2)

    conf = multilabel_confusion_matrix(y_true, y_pred)

    tn = conf[:, 0, 0]
    tp = conf[:, 1, 1]
    fn = conf[:, 1, 0]
    fp = conf[:, 0, 1]
    specificity = tp / (tp + fn)
    miss = fn / (fn + tp)

    return report

def save_model(model, filename):
    name = os.path.join('models/', filename)
    joblib.dump(model, name)


if __name__ == '__main__':
    src_dir = 'data/raw/images'
    dest_dir = 'data/processed/images'
    ann_dir = 'data/raw/annotations'
    fname = 'instances_default.json'

    # print('Fetching & resizing images...')
    # image_pre = ImagePreprocessor(src_dir, dest_dir)
    # image_set = image_pre.resize_save(save=True, color=True)

    print('Making X & y...')
    lemons = DatasetBuilder(ann_dir, fname)
    X, y = lemons.load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y)#, test_size=0.25)

    # l1s = [0.01, 0.1, 0.5, 0.9, 0.99]
    # regr = LogisticRegressionCV(class_weight='balanced',
    #                            max_iter=1000,
    #                            tol=0.001,
    #                            n_jobs=-1,
    #                            penalty='elasticnet',
    #                            l1_ratios=l1s,
    #                            multi_class='multinomial',
    #                            solver='saga',
    #                            verbose=1)
    # print('Fitting CV model\n')
    # model = regr.fit(X_train, y_train)

    regr = LogisticRegression(class_weight='balanced',
                              max_iter=100,
                              penalty='l2',
                              tol=0.05,
                              solver='saga',
                              multi_class='multinomial',
                              n_jobs=-1,
                              verbose=1)
'''
    print('\nFitting model...\n')
    m = regr.fit(X_train, y_train)
    print(m)
    model_filename = 'log_regression_model.joblib'
    print('\nSaving model...')
    save_model(m, model_filename)

    model = joblib.load(os.path.join('models/', model_filename))
'''
    print('\nPredicting!')

    y_pred = model.predict(X_test)

    print(f'\nPredicted classes: \n{y_pred}')

    avg_type = 'weighted'
    report = get_scores(y_test, y_pred, avg_type)

    print(report)
