import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MakePlots():
    def __init__(self):
        pass


    def pca_var_plot(self, pca, color):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        plt.title('Variance plots')

        ax[0].plot(pca.explained_variance_,
                linewidth=2,
                color=color)
        ax[0].axis('tight')
        ax[0].set_xlim(0, pca.n_components_)
        ax[0].set_xlabel('n_components')
        ax[0].set_ylabel('explained_variance_')

        prop_var_expl = np.cumsum(pca.explained_variance_ratio_)

        ax[1].plot(prop_var_expl,
                linewidth=2,
                color=color,               
                label='Explained variance')
        ax[1].axhline(0.9,
                    label='90% goal',
                    linestyle='--', color="black",
                    linewidth=1)
        ax[1].set_ylabel('cumulative prop. of explained variance')
        ax[1].set_xlim(0, pca.n_components_ + 3)
        ax[1].set_xlabel('number of principal components')
        ax[1].legend(loc='lower right')


if __name__ == '__main__':
    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}

    plt.rc('font', **font)
    plot_size = (10, 10)

    chart_color = 'm'

    plots = MakePlots()
