import pandas as pd

def read_csv_with_shuffle(path, random_state):
    df = pd.read_csv(path, index_col=0, dtype=object)
    df.columns = df.columns.astype(str)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df