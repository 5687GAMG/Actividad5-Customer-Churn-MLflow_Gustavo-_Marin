
import pandas as pd

def limpiar_datos(path_input, path_output):
    df = pd.read_csv(path_input)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    df.to_csv(path_output, index=False)
    return df
