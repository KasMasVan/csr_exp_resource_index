import pandas as pd
import matplotlib.pyplot as plt

def main():
    result_path = f"./language_modeling.csv"
    df = pd.read_csv(result_path)
    print(df.head())

if __name__ == "__main__":
    main()