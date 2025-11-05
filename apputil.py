import pickle
import pandas as pd

# Load the models
with open('model_1.pickle', 'rb') as f:
    model_1 = pickle.load(f)

with open('model_2.pickle', 'rb') as f:
    model_2_data = pickle.load(f)
    model_2 = model_2_data['model']
    roast_cat = model_2_data['roast_cat']


def predict_rating(df_X):
    """
    Predicts rating values from a dataframe with 100g_USD and roast columns.
    
    """
    predictions = []
    
    for idx, row in df_X.iterrows():
        price = row['100g_USD']
        roast = row['roast']
        
        # Check if roast value is in training data
        if roast in roast_cat:
            # Use model_2 (both features)
            roast_encoded = roast_cat[roast]
            X = pd.DataFrame([[price, roast_encoded]], columns=['100g_USD', 'roast_encoded'])
            pred = model_2.predict(X)[0]
        else:
            # Use model_1 (only 100g_USD)
            X = pd.DataFrame([[price]], columns=['100g_USD'])
            pred = model_1.predict(X)[0]
        
        predictions.append(pred)
    
    return pd.Series(predictions).values# your code here
