config = {
    "root_dir": '/home/mdsamiul/github_project/fair_classifier_ml/',
    "dataset_dir": "/home/mdsamiul/github_project/fair_classifier_ml/data/adult.data",
    # "root_dir": 'E:/canada syntex/Github/fair_classifier_ml',
    # "dataset_dir": "E:/canada syntex/Github/fair_classifier_ml/data/adult.data",

    "batch_size": 128,
    "iteration": 100,
    "test_size": 0.2,
    "gpu" : "3",

    "experiment": "simpleModel"
}


def initializing():

    # Create visualization directory
    config['visualization_dir'] = config['root_dir'] + '/visualization/'
