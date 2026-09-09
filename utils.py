import pandas as pd
import os

def preprocess_and_save(filepath):
    """
    Reads a file-like object (from Flask) into a pandas DataFrame.
    Returns:
        - df (DataFrame): The processed DataFrame.
        - cols (list): List of column names.
        - df_html (str): Full HTML of the DataFrame (as requested by app.py).
        - err (str): Error message, if any.
    """
    try:
        # Assume `filepath` is actually a FileStorage object from Flask
        file = filepath  # Rename for clarity
        filename = file.filename
        _, ext = os.path.splitext(filename)
        ext = ext.lower().replace(".", "")  # normalize extension like "csv", "json", etc.

        if ext == "csv":
            df = pd.read_csv(file)
        elif ext == "json":
            df = pd.read_json(file)
        elif ext in {"xlsx", "xls"}:
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type")

        df.dropna(how='all', inplace=True)
        cols = df.columns.tolist()

        df_html = ""  # placeholder, as expected by main app.py
        err = ""
        return df, cols, df_html, err

    except Exception as e:
        return None, [], "", str(e)