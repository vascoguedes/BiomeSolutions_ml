import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def get_data():
    df = pd.read_csv('Crop_recommendation.csv')
    return df[['N', 'P', 'K', 'ph', 'label']]


def pred_best_vals(culture):
    df = get_data()
    X = df[['label']]
    y = df[['N', 'P', 'K', 'ph']]
    print(y)

    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)

    # Define a column transformer to apply the one-hot encoder to the 'color' column
    preprocessor = ColumnTransformer(
        transformers=[('onehot', onehot_encoder, ['label'])],
        remainder='passthrough')

    # Define a regression model to use for each output column
    estimator = RandomForestRegressor()

    # Create a multi-output regression model using the defined estimator and preprocessor
    multioutput = MultiOutputRegressor(estimator)

    # Train the model on the input and output data
    multioutput.fit(preprocessor.fit_transform(X), y)

    # Use the model to predict multiple output columns for new input data
    new_data = [[culture]]
    encoded_data = preprocessor.transform(
        pd.DataFrame(new_data, columns=X.columns))
    predictions = multioutput.predict(encoded_data)

    print(predictions)


pred_best_vals("maize")


def plot(t):
    df = get_data()
    # oriented to horizontal
    sns.stripplot(y="label", x=t, hue="label",
                  orient="h", data=df, size=5)
    plt.show()
