import os

def save_to_pickle(df, path, filename):
    """
    Saves a DataFrame to a pickle file, creating directories if they don't exist.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to save.
    path (str): The directory path where the file will be saved.
    filename (str): The name of the pickle file.
    """
    # Ensure the directory exists, create if it doesn't
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save the DataFrame to the specified pickle file
    file_path = os.path.join(path, filename)
    df.to_pickle(file_path)