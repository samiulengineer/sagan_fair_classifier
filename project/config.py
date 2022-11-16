from datetime import datetime


config = {
    "root_dir": '/home/mdsamiul/github_project/fair_classifier_ml/',
    "dataset_dir": "/home/mdsamiul/github_project/fair_classifier_ml/data/adult.data",
    # "root_dir": 'E:/canada syntex/Github/fair_classifier_ml',
    # "dataset_dir": "E:/canada syntex/Github/fair_classifier_ml/data/adult.data",

    "batch_size": 128,
    "epochs": 20,
    "test_size": 0.2,
    "learning_rate": 0.0003,
    "pRuleThreshold": 0.3,

    "model_name": "mlalgo",    # mlalgo=ML algorithms, clf=simple model
    "experiment": "simpleModel",
    "load_model_dir": None,
    "load_model_name": "aclf_fairML_epochs_20_13-Oct-22.hdf5",

    "csv": True,
    "val_pred_plot": True,
    "lr": True,
    "tensorboard": True,
    "early_stop": False,
    "checkpoint": True,
    "patience": 300  # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically

}


def initializing():

    # Create Callbacks paths
    config['tensorboard_log_name'] = "{}_{}_epochs_{}_{}".format(
        config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
    config['tensorboard_log_dir'] = config['root_dir'] + \
        '/logs/' + config['model_name']+'/'

    config['csv_log_name'] = "{}_{}_epochs_{}_{}.csv".format(
        config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
    config['csv_log_dir'] = config['root_dir'] + \
        '/csv_logger/' + config['model_name']+'/'

    config['checkpoint_name'] = "{}_{}_epochs_{}_{}.hdf5".format(
        config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
    config['checkpoint_dir'] = config['root_dir'] + \
        '/model/' + config['model_name']+'/'

    # Create save model directory
    if config['load_model_dir'] == None:
        config['load_model_dir'] = config['root_dir'] + \
            '/model/' + config['model_name']+'/'

    # Create Evaluation directory
    config['prediction_test_dir'] = config['root_dir'] + \
        '/prediction/' + config['model_name']+'/test/'
    config['prediction_val_dir'] = config['root_dir'] + \
        '/prediction/' + config['model_name']+'/validation/'

    # Create visualization directory
    config['visualization_dir'] = config['root_dir'] + '/visualization2/'
