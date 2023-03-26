import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import LocalOutlierFactor

from sklearn.impute import SimpleImputer


def get_data():
    df = pd.read_csv('Crop_recommendation.csv')
    return df[['N', 'P', 'K', 'ph', 'label']]


def pred_best_vals(culture):
    df = get_data()
    X = df[['label']]
    y = df[['N', 'P', 'K', 'ph']]

    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)

    # Define a column transformer to apply the one-hot encoder to the 'lable' column
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

    return predictions[0]





def get_products_needed(client_vals, culture, area):
    best = pred_best_vals(culture)
    sugestions = []

    if (client_vals[0]<best[0] and client_vals[1]<best[1] and client_vals[2]<best[2]):
        quantity = 200*area
        q=quantity
        packets = []
        if(quantity//500>0):
            packets +=[(quantity//500, 500)]
            quantity -= 500*(quantity//500)
        if(quantity//25>0):
            packets +=[(quantity//25, 25)]
            quantity -= 25*(quantity//25)
        if (quantity//10 > 0):
            packets += [(quantity//10, 10)]
            quantity -= 10*(quantity//10)
        if (quantity//1 > 0):
            packets += [(quantity//1, 1)]
            quantity -= 1*(quantity//1)
        
        sugestions += [("Biopron", q, "Kg", packets, "https://probelte.com/wp-content/uploads/2022/02/Probelte_Product_catalogue.pdf")]

    elif(client_vals[0] < best[0] and client_vals[2] < best[2]):
        quantity = 5*area
        q = quantity//20
        if q*20<quantity:
            q+=1
        packets = [(q,20)]
        sugestions += [("Vitasoil", quantity, "L", packets,
                        "https://symborg.com/pt/biofertilizantes/vitasoil/")]

    else:
        if (client_vals[1] < best[1]):
            quantity = 6*area
            q = quantity//10
            if q*10<quantity:
                q+=1
            packets = [(q,10)]
            sugestions += [("Kipant AllGrip", quantity, "L",
                            "https://www.asfertglobal.com/produtos/kiplant-allgrip-pt/")]

        if (client_vals[0] < best[0]):
            quantity = 6*area
            q = quantity//10
            if q*10 < quantity:
                q += 1
            packets = [(q, 10)]
            sugestions += [("Kiplant iNmass", quantity, "L",
                            "https://www.asfertglobal.com/produtos/kiplant-inmass-pt/")]
    
    print(sugestions)
    return sugestions


def detect_outliers(row_index):
    # Load the data into a DataFrame
    df = get_data()
    cat_col = 'label'

    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
    one_hot_data = onehot_encoder.fit_transform(df[['label']])

    # Define a column transformer to apply the one-hot encoder to the 'color' column
    preprocessor = ColumnTransformer(
        transformers=[('onehot', onehot_encoder, ['label'])],
        remainder='passthrough')

    # Convert the one-hot encoded data to a DataFrame

    one_hot_cols = [
        f'{cat_col}_{val}' for val in onehot_encoder.categories_[0]]
    one_hot_df = pd.DataFrame(one_hot_data, columns=one_hot_cols)

    df = pd.concat([df.drop(columns=[cat_col]), one_hot_df], axis=1)

    # Select the row to analyze
    row = df.loc[row_index]

    # Define the imputer transformer
    imputer = SimpleImputer(strategy='mean')

    # Fit the imputer on the data
    imputer.fit(df)

    # Fill in the missing values in the DataFrame
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)

    # Define the local outlier factor model
    model = LocalOutlierFactor(
        n_neighbors=20, contamination=0.05, novelty=False)

    # Fit the model on the data
    scores = model.fit_predict(df)

    # Get the anomaly score of the row
    original_score = scores[row_index]

    # Get the scores of all rows that are not the analyzed row
    other_scores = scores[scores != original_score]

    # Get the median score of the other rows
    median_score = np.median(other_scores)

    # Get the difference between the original score and the median score
    score_diff = original_score - median_score

    # Get the columns that contribute to the score difference
    culprit_cols = set()
    for col in df.columns:
        # Create a copy of the dataframe with the analyzed row removed
        no_row_df = df.drop(index=row_index)
        # Create a copy of the dataframe with the analyzed row's value in the current column changed
        new_row_df = no_row_df.copy()
        new_row_df.loc[row_index, col] = row[col] + 1
        # Calculate the scores for both dataframes
        no_row_scores = model.fit_predict(no_row_df)
        new_row_scores = model.fit_predict(new_row_df)
        # Get the difference between the two scores
        col_score_diff = new_row_scores[row_index] - no_row_scores[row_index]
        # If the difference is greater than the score difference, the column is a culprit
        if col_score_diff > score_diff:
            culprit_cols.add(col)

    if len(culprit_cols) > 0:
        print(
            f"The row contains outliers in columns: {', '.join(culprit_cols)}")
    else:
        print("The row does not contain outliers.")


#detect_outliers(0)
