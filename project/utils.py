
import os
import math
import pathlib
import tensorflow as tf
from tensorflow import keras

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
    print(config['csv_log_dir'])
    print(config['tensorboard_log_dir'])
    if test:
        pathlib.Path(config['prediction_test_dir']).mkdir(
            parents=True, exist_ok=True)
    else:
        pathlib.Path(config['csv_log_dir']
                     ).mkdir(parents=True, exist_ok=True)
        pathlib.Path(config['tensorboard_log_dir']).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(config['checkpoint_dir']).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(config['prediction_val_dir']).mkdir(
            parents=True, exist_ok=True)
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
