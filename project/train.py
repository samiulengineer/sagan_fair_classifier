import time
import os
import tensorflow as tf
from model import get_model
from metrics import get_metrics
from dataset import get_dataloader
from utils import create_paths, SelectCallbacks
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from config import config, initializing
initializing()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# creating directory
create_paths()

X_train, y_train, X_val, y_val, Z_train, Z_val = get_dataloader()

n_features = X_train.shape[1]
# n_sensitive = Z_train.shape(1)


model = get_model(n_features)

metrics = list(get_metrics().values())

# Callbacks
# ----------------------------------------------------------------------------------------------
loggers = SelectCallbacks((X_val, y_val), model)

if config["model_name"] == "mlalgo":
    t0 = time.time()
    accuracy = []
    print("Model name \t\t Accuracy")
    for i in model.values():
        mo = i
        mo.fit(X_train, y_train)
        mo_pred = mo.predict(X_val)
        acc = accuracy_score(y_val, mo_pred)
        accuracy.append(acc)
        print("{}\t{:.2f}%".format(i, acc*100))
    print("training time minute: {}".format((time.time()-t0)/60))

else:
    # training deep models

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=metrics)
    print(model.summary())

    # train on train set
    t0 = time.time()
    history = model.fit(X_train, y_train,
                        epochs=config["epochs"],
                        verbose=1,
                        validation_data=(X_val, y_val),
                        shuffle=False,
                        callbacks=loggers.get_callbacks((X_val, y_val), model)
                        )
    print("training time minute: {}".format((time.time()-t0)/60))
