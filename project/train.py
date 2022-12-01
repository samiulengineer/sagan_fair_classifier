import os
import gc
import argparse
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import load_ICU_data
from model import FairClassifier
from utils import create_paths
from config import config, initializing
initializing()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.experimental.list_physical_devices('gpu')


# Parsing variable ctrl + /
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--iteration", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--test_size", type=float)
parser.add_argument("--gpu")

args = parser.parse_args()


# Set up train configaration
# ----------------------------------------------------------------------------------------------
args = vars(args)
for key in args.keys():
        if args[key] != None:
            config[key] = args[key]


# creating directory
# ----------------------------------------------------------------------------------------------
create_paths()


# training 
# ----------------------------------------------------------------------------------------------
# threshold = [0.1, 0.5, 1.5, 3.1, 4.5, 5.5, 10.5]
# threshold = np.round(np.arange(0.1, 10, 0.5), 2)
threshold = [0.1]
models = []

# prev_mem = 0
# first_mem = 0

for i in threshold:
        print("---------------------------")
        print("For Attention Module:", i)
        
        X_proxy, X_real,  y, Z = load_ICU_data(config['dataset_dir'], i)

        # split into train/test set for proxy features
        X_train_proxy, X_test_proxy, y_train_proxy, y_test_proxy, Z_train_proxy, Z_test_proxy = train_test_split(X_proxy, y, Z, 
                                                                             test_size=config['test_size'],
                                                                             stratify=y, random_state=7)
        
        
        # split into train/test set for real features
        X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X_real, y, Z,
                                                                             test_size=config['test_size'],
                                                                             stratify=y, random_state=7)
        

        # standardize the real data
        scaler = StandardScaler().fit(X_train)
        scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        X_train = X_train.pipe(scale_df, scaler) 
        X_test = X_test.pipe(scale_df, scaler) 
        
        # standardize the proxy data
        scaler = StandardScaler().fit(X_train_proxy)
        scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        X_train_proxy = X_train_proxy.pipe(scale_df, scaler) 
        X_test_proxy = X_test_proxy.pipe(scale_df, scaler)
        


        # initialise FairClassifier
        clf = FairClassifier(n_features=X_train.shape[1], n_features_proxy=X_train_proxy.shape[1], n_sensitive=Z_train.shape[1],
                                lambdas=[5., 5.])
        

        # pre-train both adverserial and classifier networks    X_train_proxy, y_train_proxy, Z_train_proxy,
        clf.pretrain(X_train, y_train, Z_train, X_train_proxy, y_train_proxy, Z_train_proxy, verbose=0, epochs=5)
        
        clf.fit(X_train_proxy, y_train_proxy, Z_train_proxy, i,
                validation_data=(X_test, y_test, Z_test),
                T_iter=config["iteration"], batch_size=config["batch_size"],
                save_figs=True, verbose=1)
        
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