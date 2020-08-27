import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.decomposition import PCA

import sys
sys.path.append('./src')
# from img_proc import ImagePreprocessor
from dataset_prep import AnnotationsParser, DatasetBuilder
from eda import MakePlots


if __name__ == '__main__':
    ann_dir = '/home/chris/Dropbox/galvanize/capstones/lemon-defect-detection/data/raw/annotations'
    fname = 'instances_default.json'

    lemons = DatasetBuilder(ann_dir, fname)
    X, y = lemons.load_data()

    # print('\nDoing PCA...')
    # pca = PCA(n_components=0.9)  
    # X_pca = pca.fit_transform(X)
    # print(f'Number of components covering 90% variance: {pca.n_components_}')
    # print(f'Variance components: {np.around(pca.explained_variance_[0:6], 1)}')

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

    # pipeline = Pipeline()

    regr = LogisticRegression(class_weight='balanced',
                              max_iter=500,
                              penalty='elasticnet',
                              l1_ratio=0.1,
                              solver='saga',
                              multi_class='multinomial',
                              n_jobs=-1)

    print('\nFitting model...\n')
    regr.fit(X_train, y_train)

    probs = regr.predict_proba(X_test)
    pred_classes = regr.predict(X_test)
    r2 = r2_score(y_test, pred_classes, multioutput='raw_values')

    # print(f'Probabilities: {probs}')
    print(f'Predicted classes: {pred_classes}')
    # print(f'ROC: {roc_auc_score(y_test, y_hat)}')
    print(f'R2 score: {r2}')
    # print(f'Mean Squared Error: {round(mse, 2)}')
    # print(f'RMSE: {round(rmse, 2)}')

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)

    # ax.plot(mse_list, c='k', label='MSE')
    ax.scatter(pred_classes, y_test, c=np.random(1, 2, size=len(y_test)), label='predicted classes')
    plt.legend()
    plt.show()
