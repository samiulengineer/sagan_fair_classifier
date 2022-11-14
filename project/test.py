import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset import get_dataloader
from metrics import get_metrics, p_rule
from config import config, initializing
initializing()


# X_test, y_test, Z_test = get_dataloader(test=True)
test_dataset, Z_test = get_dataloader(test=True)

model = load_model(os.path.join(
    config['load_model_dir'], config['load_model_name']), compile=False)


# model evaluate
metrics = list(get_metrics().values())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
model.evaluate(test_dataset)


# prediction

# y_pred = pd.Series(model.predict(X_test.values).ravel(), index=y_test.index)
# y_pred = model.predict(test_dataset)

# print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")
# print(f"Accuracy: {100*accuracy_score(y_test, (y_pred>0.5)):.1f}%")

# # 80% rule to calculate fairness in model
# print("The classifier satisfies the following %p-rules:")
# print(f"\tgiven attribute race; {p_rule(y_pred, Z_test['race']):.0f}%-rule")
# print(f"\tgiven attribute sex;  {p_rule(y_pred, Z_test['sex']):.0f}%-rule")
# print(np.array(y_pred).shape)
