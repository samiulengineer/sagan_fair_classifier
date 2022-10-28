from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

from config import config, initializing

initializing()


def mlAlgo(n_features):
    model = {"Logistic Regression": LogisticRegression(),
             "KNN": KNeighborsClassifier(n_neighbors=50),
             "Decision tree": DecisionTreeClassifier(),
             "Naive bayes": GaussianNB(),
             "Linear SVM": SVC(kernel='linear'),
             "Gaussian SVM": SVC(kernel='rbf'),
             "Random forest": RandomForestClassifier(n_estimators=100)}

    return model


def clf(n_features):
    inputs = Input(shape=(n_features,))
    dense1 = Dense(32, activation='relu')(inputs)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    dense3 = Dense(32, activation="relu")(dropout2)
    dropout3 = Dropout(0.2)(dense3)
    outputs = Dense(1, activation='sigmoid')(dropout3)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def get_model(n_features):
    """
    Summary:
        create new model object for training
    Arguments:
        config (dict): Configuration directory
    Return:
        model (object): keras.Model class object
    """

    models = {'clf': clf,
              'mlalgo': mlAlgo
              }

    return models[config["model_name"]](n_features)


if __name__ == '__main__':

    model = get_model(94)
    # model.summary()
    print(type(model))
