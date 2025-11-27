
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data():
    input_path = "../aapl.us_raw.txt"  # lokasi dataset mentah
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)
    return df

def clean_data(df):
    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop NA Date rows
    df = df.dropna(subset=["Date"])

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def add_time_features(df):
    df["day_sin"] = np.sin(2 * np.pi * df["Date"].dt.day / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["Date"].dt.day / 31)

    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)

    return df

def normalize(df):
    scaler = MinMaxScaler()

    df["Close_norm"] = scaler.fit_transform(df[["Close"]])

    return df

def save_output(df):
    output_path = "aapl.us.txt_preprocessing.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed dataset to {output_path}")

def main():
    df = load_data()
    df = clean_data(df)
    df = add_time_features(df)
    df = normalize(df)
    save_output(df)

if __name__ == "__main__":
    main()

