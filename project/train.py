import os
import gc
# import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import load_ICU_data
from model import FairClassifier
from utils import plot_curve, create_paths
from config import config, initializing
initializing()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.experimental.list_physical_devices('gpu')

# process = psutil.Process(os.getpid())
# tf.config.run_functions_eagerly(True)  # to use .numpy()

# creating directory
create_paths()

# threshold = [0.1, 0.5, 1.5, 3.1, 4.5, 5.5, 10.5]
# threshold = np.round(np.arange(0.1, 10, 0.5), 2)
threshold = [0.1]
models = []

# prev_mem = 0
# first_mem = 0

for i in threshold:
        print("---------------------------")
        print("For Attention Module:", i)
        X, y, Z = load_ICU_data(config['dataset_dir'], i)

        # split into train/test set
        X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, 
                                                                             test_size=config['test_size'],
                                                                             stratify=y, random_state=7)

        # standardize the data
        scaler = StandardScaler().fit(X_train)
        scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        X_train = X_train.pipe(scale_df, scaler) 
        X_test = X_test.pipe(scale_df, scaler) 

        # initialise FairClassifier
        clf = FairClassifier(n_features=X_train.shape[1], n_sensitive=Z_train.shape[1],
                                lambdas=[5., 5.])

        # pre-train both adverserial and classifier networks
        clf.pretrain(X_train, y_train, Z_train, verbose=0, epochs=5)
        
        clf.fit(X_train, y_train, Z_train, i,
                validation_data=(X_test, y_test, Z_test),
                T_iter=300, save_figs=True, verbose=1)
        
        models.append(clf)
        
        # clean up
        _ = gc.collect()
        keras.backend.clear_session()
        
        # # show memory usage
        # mem = process.memory_info().rss
        # if i == 0:
        #         first_mem = mem
        # print(
        #         f"iteration {i}: rss {mem >> 20} MB ({(mem - prev_mem) >> 10:+} KB; "
        #         + f"{((mem - first_mem) // max(1, i)) >> 10:+} KB/it.)"
        # )
        # prev_mem = mem


# Attention Module vs P% rule curve plot
# ----------------------------------------------------------------------------------------------
# p_race = []
# p_sex = []
# for i, j in zip(models, threshold):
#     p_race.append(i._fairness_metrics.loc[40, 'race'])
#     p_sex.append(i._fairness_metrics.loc[40, 'sex'])
    
# fname = config['visualization_dir'] + 'P_ruleVsAttention.jpg'
# ylabel = "P%-rule"
# plot_curve(threshold, p_race, p_sex, fname, ylabel)


# ----------------------------------------------------------------------------------------------

# X_train, y_train, X_val, y_val, Z_train, Z_val = get_dataloader()
# n_features = X_train.shape[1]
# # n_sensitive = Z_train.shape(1)
# model = get_model(n_features)
# metrics = list(get_metrics().values())
# # Callbacks
# # ----------------------------------------------------------------------------------------------
# loggers = SelectCallbacks((X_val, y_val), model)
# # ----------------------------------------------------------------------------------------------
# if config["model_name"] == "mlalgo":
#     # training machine learning models
#     t0 = time.time()
#     accuracy = []
#     print("Model name \t\t Accuracy")
#     for i in model.values():
#         mo = i
#         mo.fit(X_train, y_train)
#         mo_pred = mo.predict(X_val)
#         acc = accuracy_score(y_val, mo_pred)
#         accuracy.append(acc)
#         print("{}\t{:.2f}%".format(i, acc*100))
#     print("training time minute: {}".format((time.time()-t0)/60))

# else:
#     # training deep models

#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam', metrics=metrics)
#     print(model.summary())

#     # train on train set
#     t0 = time.time()
#     history = model.fit(X_train, y_train,
#                         epochs=config["epochs"],
#                         verbose=1,
#                         validation_data=(X_val, y_val),
#                         shuffle=False,
#                         callbacks=loggers.get_callbacks((X_val, y_val), model)
#                         )
#     print("training time minute: {}".format((time.time()-t0)/60))
