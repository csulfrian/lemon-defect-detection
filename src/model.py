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
import sys
sys.path.append('./src')
import numpy as np
import pandas as pd


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

    return recall, precision, accuracy, f1, jaccard, specificity, miss, conf, report

def save_model(model):
    with open('models/sobel_lr.txt', 'wb') as fh:
            pickle.dump(model, fh)


if __name__ == '__main__':
    src_dir = 'data/raw/images'
    dest_dir = 'data/processed/images'
    ann_dir = 'data/raw/annotations'
    fname = 'instances_default.json'

    image_pre = ImagePreprocessor(src_dir, dest_dir)
    image_set = image_pre.resize_save(save=False, color=True)

    lemons = DatasetBuilder(ann_dir, fname)
    X, y = lemons.load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y)#, test_size=0.25)

    l1s = [0.01, 0.1, 0.5, 0.9, 0.99]
    regr = LogisticRegressionCV(class_weight='balanced',
                               max_iter=1000,
                               tol=0.001,
                               n_jobs=-1,
                               penalty='elasticnet',
                               l1_ratios=l1s,
                               multi_class='multinomial',
                               solver='saga',
                               verbose=1)
    print('Fitting CV model\n')
    model = regr.fit(X_train, y_train)
    save_model(model)

    # regr = LogisticRegression(class_weight='balanced',
    #                           max_iter=1000,
    #                           penalty='l2',
    #                           tol=0.0005,
    #                           solver='saga',
    #                           multi_class='multinomial',
    #                           n_jobs=-1,
    #                           verbose=1)

    # print('\nFitting model...\n')
    # regr.fit(X_train, y_train)

    print('\nPredicting!')

    y_pred = model.predict(X_test)

    print(f'\nPredicted classes: \n{y_pred}')

    avg_type = 'weighted'
    recall, precision, accuracy, f1, jaccard, specificity, miss, conf, report = get_scores(y_test, y_pred, avg_type)

    print(f'Recall: {recall:0.2f}')
    print(f'Precision: {precision:0.2f}')
    print(f'Accuracy: {accuracy:0.2f}')
    print(f'F1 score: {f1:0.2f}')
    print(f'Jaccard Score: {jaccard}')
    print(f'Specificity (True Negative rate): {specificity}')
    print(f'Misses (False Negative rate): {miss}')
    print(f'Confusion matrix: \n{conf}')
    print(report)
