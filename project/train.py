import time
import os
import tensorflow as tf
from model import get_model
from metrics import get_metrics
from dataset import get_dataloader
from utils import create_paths, SelectCallbacks
from config import config, initializing
initializing()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# creating directory
create_paths(test=False)

train_dataset, Z_train_dataset, val_dataset, Z_val_dataset, class_weights = get_dataloader()

n_features = 0
n_sensitive = 0
for x, y in train_dataset:
    n_features = int(tf.shape(x)[1])
    break

for z in Z_train_dataset:
    n_sensitive = int(tf.shape(z)[1])
    break


# to see shape
# dataset_to_numpy = list(train_dataset.as_numpy_iterator())
# shape = tf.shape(dataset_to_numpy)
# print(shape)


model = get_model(n_features, n_sensitive)

metrics = list(get_metrics().values())

# Callbacks
# ----------------------------------------------------------------------------------------------
loggers = SelectCallbacks(val_dataset, model)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
print(model.summary())

# train on train set
t0 = time.time()
history = model.fit(train_dataset,
                    epochs=config["epochs"],
                    verbose=1,
                    validation_data=val_dataset,
                    shuffle=False,
                    class_weight=class_weights,
                    callbacks=loggers.get_callbacks(val_dataset, model))
print("training time minute: {}".format((time.time()-t0)/60))
