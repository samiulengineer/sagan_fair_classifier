import numpy as np
import pandas as pd
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette=[sns.color_palette('muted')[i] for i in [0,2]], 
        color_codes=True, context="talk")
from IPython import display
from scipy.interpolate import make_interp_spline

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

from utils import plot_distributions
from metrics import p_rule, fairness_metrics
from config import config, initializing

initializing()

# Fari classifier
#------------------------------------------------------------------------------------------------------
class FairClassifier(object):
    """
    Summary:
        fair classifier
    Arguments:
        n_features (int):  number of features
        n_sensitive (int): number of sensitive features
        lambdas (float): loss weight
    Return:
        prediction
    """
    
    def __init__(self, n_features, n_sensitive, lambdas):
        self.lambdas = lambdas
        self.n_features = n_features

        clf_inputs = Input(shape=(n_features,))
        adv_inputs = Input(shape=(1,))
        
        clf_net = self._create_clf_net(clf_inputs)
        adv_net = self._create_adv_net(adv_inputs, n_sensitive)
        #print(adv_net.summary())
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(adv_net)

        # compile model. Three model compiletion: clf, clf_w_adv and adv
        self._clf = self._compile_clf(clf_net)
        self._clf_w_adv = self._compile_clf_w_adv(clf_inputs, clf_net, adv_net)
        self._adv = self._compile_adv(clf_inputs, clf_net, adv_net, n_sensitive)
        # print(self._adv.summary())

        self._val_metrics = None
        self._fairness_metrics = None
        self.fm_metrics = None
        # self.metrics_res = pd.DataFrame()

        self.predict = self._clf.predict
        
    # def attention(input, target, sen_attr):
        
    #     return X, y, Z

    # making all layer trainable
    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag
        return make_trainable

    # construct model ----------------------------------------------------------

    # clf net    input layer + 3 hidden layer + output layer
    def _create_clf_net(self, inputs):
        dense1 = Dense(32, activation='relu')(inputs)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation='relu')(dropout2)
        dropout3 = Dropout(0.2)(dense3)
        outputs = Dense(1, activation='sigmoid', name='y')(dropout3)
        return Model(inputs=[inputs], outputs=[outputs])
    
    
    # adv net    input layer + 3 hidden layer + output layer * n_sensitive
    def _create_adv_net(self, inputs, n_sensitive):
        dense1 = Dense(32, activation='relu')(inputs)
        dense2 = Dense(32, activation='relu')(dense1)
        dense3 = Dense(32, activation='relu')(dense2)
        outputs = [Dense(1, activation='sigmoid')(dense3) for _ in range(n_sensitive)]
        return Model(inputs=[inputs], outputs=outputs)
    
    # compile  -----------------------------------------------------------------

    # compile clf
    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        return clf
        
    # compile clf_w_adv
    def _compile_clf_w_adv(self, inputs, clf_net, adv_net):
        clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs)]+adv_net(clf_net(inputs))) # outputs=[clf_net(inputs)]+adv_net(clf_net(inputs))
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.]+[-lambda_param for lambda_param in self.lambdas] # loss [1.0, -5.0, -5.0]
        clf_w_adv.compile(loss=['binary_crossentropy']*(len(loss_weights)), 
                          loss_weights=loss_weights,  # adding loss weight
                          optimizer='adam')
        return clf_w_adv

    # compile adv
    def _compile_adv(self, inputs, clf_net, adv_net, n_sensitive):
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs))) # outputs=adv_net(clf_net(inputs))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        adv.compile(loss=['binary_crossentropy']*n_sensitive, loss_weights=self.lambdas,  # added loss weights and loss=['binary_crossentropy']*n_sensitive
                    optimizer='adam')
        return adv

    # compute weights based on features
    def _compute_class_weights(self, data_set, classes=[0, 1]):
        class_weights = []
        if len(data_set.shape) == 1: 
            # for single feature
            balanced_weights = compute_class_weight('balanced', classes=classes, y=data_set)
            class_weights.append(dict(zip(classes, balanced_weights)))
        else:
            # for multiple feature
            n_attr =  data_set.shape[1]
            for attr_idx in range(n_attr):
                balanced_weights = compute_class_weight('balanced', classes=classes,
                                                        y=np.array(data_set)[:,attr_idx])
                class_weights.append(dict(zip(classes, balanced_weights)))
        return class_weights          
    
    # compute weights based on targets
    def _compute_target_class_weights(self, y, classes=[0, 1]):
        balanced_weights =  compute_class_weight('balanced', classes=classes, y=y)
        class_weights = {'y': dict(zip(classes, balanced_weights))}
        return class_weights
        
    
    def pretrain(self, x, y, z, epochs=10, verbose=0):
        self._trainable_clf_net(True)
        self._clf.fit(x.values, y.values, epochs=epochs, verbose=verbose) # training clf
        self._trainable_clf_net(False)

        self._trainable_adv_net(True)
        class_weight_adv = self._compute_class_weights(z)
        # passing z value instead of y and split the z into two
        self._adv.fit(x.values, np.hsplit(z.values, z.shape[1]), class_weight=class_weight_adv, # Split an array into multiple sub-arrays horizontally (column-wise)
                      epochs=epochs, verbose=verbose) # training adv
        
    
    def fit(self, x, y, z, atten_wei, validation_data=None, T_iter=250, batch_size=128, save_figs=False, verbose=0):
        
        n_sensitive = z.shape[1]
        if validation_data is not None:
            x_val, y_val, z_val = validation_data
        
        class_weight_clf = [{0:1., 1:1}]
        class_weight_adv = self._compute_class_weights(z)
        class_weight_clf_w_adv = class_weight_clf + class_weight_adv
        self._val_metrics = pd.DataFrame()
        self._fairness_metrics = pd.DataFrame()
        self.fm_metrics = pd.DataFrame(columns=z_val.columns)  
        
        pathlib.Path(config['visualization_dir'] + f'output_{atten_wei}_{self.n_features}').mkdir(parents=True, exist_ok=True)
        for idx in range(T_iter):
            if validation_data is not None:
                y_pred = pd.Series(self._clf.predict(x_val.values).ravel(), index=y_val.index)
                self._val_metrics.loc[idx, 'ROC AUC'] = roc_auc_score(y_val, y_pred)
                self._val_metrics.loc[idx, 'Accuracy'] = (accuracy_score(y_val, (y_pred>0.5))*100)
                
                for sensitive_attr in z_val.columns:
                    self._fairness_metrics.loc[idx, sensitive_attr] = p_rule(y_pred, z_val[sensitive_attr])

                    self.fm_metrics[sensitive_attr] = self.fm_metrics[sensitive_attr].astype(object)    # by default pandas do not except object 
                    self.fm_metrics.loc[idx, sensitive_attr] = fairness_metrics(y_val.values, (y_pred>0.5).values, z_val[sensitive_attr].values)


                display.clear_output(wait=True)
                plot_distributions(y_pred, z_val, atten_wei, idx+1, self._val_metrics.loc[idx],
                                   self._fairness_metrics.loc[idx], 
                                   self.fm_metrics.loc[idx],
                                   fname = config['visualization_dir'] + f'output_{atten_wei}_{self.n_features}/{idx+1:08d}.jpg' if save_figs else None)
                # plt.show(plt.gcf())
            
            
            # train adverserial
            self._trainable_clf_net(False)
            self._trainable_adv_net(True)
            self._adv.fit(x.values, np.hsplit(z.values, z.shape[1]), batch_size=batch_size,
                          class_weight=class_weight_adv, epochs=1, verbose=verbose)
            
            # train clf_w_adv.fit 
            # !Changed this into several epochs on whole dataset instead of single random minibatch!
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x))[:batch_size]
            self._clf_w_adv.fit(x.values, [y.values]+np.hsplit(z.values, z.shape[1]), batch_size=len(x), 
                                class_weight=class_weight_clf_w_adv, epochs=5, verbose=verbose)

