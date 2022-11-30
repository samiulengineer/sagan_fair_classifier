
import os
import math
import pathlib
import numpy as np
np.random.seed(7)
from tensorflow import keras

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette=[sns.color_palette('muted')[i] for i in [0,2]], 
        color_codes=True, context="talk")
from scipy.interpolate import make_interp_spline
#%matplotlib inline

from config import config, initializing
initializing()


def create_paths(test=False):
    """
    Summary:
        creating paths for train and test if not exists
    Arguments:
        config (dict): configuration dictionary
        test (bool): boolean variable for test directory create
    Return:
        create directories
    """
    pathlib.Path(config['visualization_dir']).mkdir(parents=True, exist_ok=True)
    # if test:
    #     pathlib.Path(config['prediction_test_dir']).mkdir(
    #         parents=True, exist_ok=True)
    # else:
    #     pathlib.Path(config['csv_log_dir']
    #                  ).mkdir(parents=True, exist_ok=True)
    #     pathlib.Path(config['tensorboard_log_dir']).mkdir(
    #         parents=True, exist_ok=True)
    #     pathlib.Path(config['checkpoint_dir']).mkdir(
    #         parents=True, exist_ok=True)
    #     pathlib.Path(config['prediction_val_dir']).mkdir(
    #         parents=True, exist_ok=True)
    
    
# Callbacks and Prediction during Training
# ----------------------------------------------------------------------------------------------

class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self, val_dataset, model):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
            config (dict): configuration dictionary
        Return:
            class object
        """
        super(keras.callbacks.Callback, self).__init__()

        self.val_dataset = val_dataset
        self.model = model
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """
        drop = 0.5
        epoch_drop = config['epochs'] / 8.
        lr = config['learning_rate'] * \
            math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr

    def on_epoch_end(self, epoch, logs={}):
        """
        Summary:
            call after every epoch to predict mask
        Arguments:
            epoch (int): current epoch
        Output:
            save predict mask
        """
        print("end epoch")

    def get_callbacks(self, val_dataset, model):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """
        if config['csv']:  # save all type of accuracy in a csv file for each epoch
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(
                config['csv_log_dir'], config['csv_log_name']), separator=",", append=False))

        if config['checkpoint']:  # save the best model
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(
                config['checkpoint_dir'], config['checkpoint_name']), save_best_only=True))

        if config['tensorboard']:  # Enable visualizations for TensorBoard
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(
                config['tensorboard_log_dir'], config['tensorboard_log_name'])))

        if config['lr']:  # adding learning rate scheduler
            self.callbacks.append(
                keras.callbacks.LearningRateScheduler(schedule=self.lr_scheduler))

        # if config['early_stop']:  # early stop the training if there is no change in loss
        #     self.callbacks.append(keras.callbacks.EarlyStopping(
        #         monitor='my_mean_iou', patience=config['patience']))

        # if config['val_pred_plot']:  # plot validated image for each epoch
        #     self.callbacks.append(SelectCallbacks(
        #         val_dataset, model, config))

        return self.callbacks


# Plot during training
# ----------------------------------------------------------------------------------------------

def plot_distributions(y, Z, atten_wei, iteration=None, val_metrics=None, p_rules=None, fm= None, fname=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    legend={'race': ['black','white'],
            'sex': ['female','male']}
    for idx, attr in enumerate(Z.columns):
        for attr_val in [0, 1]:
            ax = sns.kdeplot(data=y[Z[attr] == attr_val],
                             label='{}'.format(legend[attr][attr_val]), 
                             ax=axes[idx], fill=True)
        ax.set_xlim(0,1)
        ax.set_ylim(0,7)
        ax.set_yticks([])
        ax.set_title("sensitive attibute: {}".format(attr))
        if idx == 0:
            ax.set_ylabel('prediction distribution')
        ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(attr))
    if iteration:
        fig.text(1.0, 0.9, f"Training iteration #{iteration} \nAttention weight: {atten_wei}", fontsize='15')
    if val_metrics is not None:
        fig.text(1.0, 0.65, '\n'.join(["Prediction performance:",
                                       f"- ROC AUC: {val_metrics['ROC AUC']:.2f}",
                                       f"- Accuracy: {val_metrics['Accuracy']:.1f}"
                                       ]),
                                       fontsize='14')
    # if p_rules is not None:
    #     fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
    #                                  [f"- {attr}: {p_rules[attr]:.0f}%-rule" 
    #                                   for attr in p_rules.keys()]), 
    #              fontsize='16')

    # if fm is not None:
    #     fig.text(1.0, 0.2, '\n'.join(["FM:"] +
    #                                  [f"- {attr}: {fm[attr]}" 
    #                                   for attr in fm.keys()]), 
    #              fontsize='16')
    
    fig.text(1.0, 0.4,
             '\n'+ 'Race:  P%rule={:.0f}%'.format(p_rules['race']) + 
            #  '\n' + '           EOp={:.1f}'.format(fm['race'][0]) +
            #  '\n' + '           EO={:.1f}'.format(fm['race'][1]) +
             '\n' + '           DI={:.1f}'.format(fm['race'][3]) ,
             fontsize='14')
    
    fig.text(1.0, 0.15,
             '\n'+ 'Sex:  P%rule={:.0f}%'.format(p_rules['sex']) + 
            #  '\n' + '         EOp={:.1f}'.format(fm['sex'][0]) +
            #  '\n' + '         EO={:.1f}'.format(fm['sex'][1]) +
             '\n' + '         DI={:.1f}'.format(fm['sex'][3]),
             fontsize='14')
    
        
    fig.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight',  dpi = 800)
    return fig


# Plot curve
# ----------------------------------------------------------------------------------------------

def plot_curve(threshold, p_race, p_sex, fname, ylable):
    
    x1 = np.array(threshold)
    x2 = np.array(threshold)
    y1 = np.array(p_race)
    y2 = np.array(p_sex)

    plt.figure(figsize=(15, 5))
    plt.title("Training for different Attention weight")

    plt.plot(x1, y1, '-', label='Race')
    plt.plot(x2, y2, '--', label='Sex')
    plt.legend(loc='best')

    plt.xlabel("Attention weight")
    plt.ylabel(ylable)
    # plt.xticks(ticks=threshold,labels=threshold)
    plt.savefig(fname, bbox_inches='tight',  dpi = 1000)
    plt.show()
    

# Plot smooth curve
# ----------------------------------------------------------------------------------------------

def plot_smooth_curve(threshold, p_race, p_sex, fname, ylable):
    
    x1 = np.array(threshold)
    x2 = np.array(threshold)
    y1 = np.array(p_race)
    y2 = np.array(p_sex)

    # smoothing graph
    X_Y_1 = make_interp_spline(x1, y1)
    X_Y_2 = make_interp_spline(x2, y2)

    # Returns evenly spaced numbers
    # over a specified interval.
    X_1 = np.linspace(x1.min(), x1.max(), 500)
    Y_1 = X_Y_1(X_1)

    X_2 = np.linspace(x2.min(), x2.max(), 500)
    Y_2 = X_Y_2(X_2)

    plt.figure(figsize=(15, 5))
    plt.title("Training for different Attention weight")

    plt.plot(X_1, Y_1, '-', label='Race')
    plt.plot(X_2, Y_2, '--', label='Sex')
    plt.legend(loc='best')

    plt.xlabel("Attention weight")
    plt.ylabel(ylable)
    # plt.xticks(ticks=threshold,labels=threshold)
    plt.savefig(fname, bbox_inches='tight',  dpi = 800)
    plt.show()