import os
import pathlib
import string
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import seaborn as sns

from config import config, initializing
from dataset import get_dataloader, read_data
from metrics import p_rule

sns.set(style="white", palette=[sns.color_palette('muted')[i] for i in [0, 2]],
        color_codes=True, context="talk")
initializing()


def plot_correlation(dataset, fname=None):
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 15))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
    plt.title('Dataset Correlation', loc='left', pad=20, fontsize=15)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight',  dpi=300)


def plot_distributions(y, Z, iteration=None, val_metrics=None, p_rules=None, fname=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    legend = {'race': ['black', 'white'],
              'sex': ['female', 'male']}
    for idx, attr in enumerate(Z.columns):
        for attr_val in [0, 1]:
            ax = sns.kdeplot(data=y[Z[attr] == attr_val],
                             label='{}'.format(legend[attr][attr_val]),
                             ax=axes[idx], fill=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 7)
        ax.set_yticks([])
        ax.set_title("sensitive attibute: {}".format(attr))
        if idx == 0:
            ax.set_ylabel('prediction distribution')
        ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(attr))
    if iteration:
        fig.text(1.0, 0.9, f"Training iteration #{iteration}", fontsize='16')
    if val_metrics is not None:
        fig.text(1.0, 0.65, '\n'.join(["Prediction performance:",
                                       f"- ROC AUC: {val_metrics['ROC AUC']:.2f}",
                                       f"- Accuracy: {val_metrics['Accuracy']:.1f}"]),
                 fontsize='16')
    if p_rules is not None:
        fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
                                     [f"- {attr}: {p_rules[attr]:.0f}%-rule"
                                      for attr in p_rules.keys()]),
                 fontsize='16')
    fig.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight',  dpi=300)
    return fig


if __name__ == '__main__':

    # creating visulization directory
    pathlib.Path(config['visualization_dir']).mkdir(
        parents=True, exist_ok=True)

    # get the dataset
    # column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    #                 'marital_status', 'occupation', 'relationship', 'race', 'sex',
    #                 'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    # input_data = (pd.read_csv(config["dataset_dir"], names=column_names,
    #                           na_values="?", sep=r'\s*,\s*', engine='python')
    #               .loc[lambda df: df['race'].isin(['White', 'Black'])])

    # for i in input_data.columns:
    #     values = input_data[i].unique()
    #     if (isinstance(values[0], str)):
    #         input_data[i] = pd.factorize(input_data[i])[0] + 1
    #     print(values)
    #input_data['Education'].replace(['Under-Graduate', 'Diploma '], [0, 1], inplace=True)
    # print(input_data.head())

    # get the prediction
    X_test, y_test, Z_test = get_dataloader(test=True)
    model = load_model(os.path.join(
        config['load_model_dir'], config['load_model_name']), compile=False)

    y_pred = pd.Series(model.predict(
        X_test.values).ravel(), index=y_test.index)

    # p_ruls
    p_rul_result = {
        'race': p_rule(y_pred, Z_test['race']),
        'sex': p_rule(y_pred, Z_test['sex'])
    }

    # # ploting correlation of dataset
    # plot_correlation(
    #     input_data, fname=config['visualization_dir'] + '/correlation.png')

    # show the distributions of the predicted
    fig = plot_distributions(y_pred, Z_test,
                             p_rules=p_rul_result,
                             fname=config['visualization_dir'] +
                             '/biased_training.png'
                             )
