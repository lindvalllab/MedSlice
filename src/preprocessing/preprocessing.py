import os
import pandas as pd

class Preprocessing:
    def __init__(self, path, note_text_column, start_index_column=None, end_index_column=None):
        """
        Initialize the Preprocessing class.
        
        Args:
            path (str): Path to a CSV file or a folder containing CSV files.
            note_text_column (str): Name of the column corresponding to note text.
            start_index_column (str, optional): Name of the column corresponding to ground truth start indexes.
            end_index_column (str, optional): Name of the column corresponding to ground truth end indexes.
        """
        self.path = path
        self.note_text_column = note_text_column
        self.start_index_column = start_index_column
        self.end_index_column = end_index_column
        
        # Validate the path
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The specified path '{self.path}' does not exist.")
        
        # If it's a folder, list all CSV files
        if os.path.isdir(self.path):
            self.files = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.csv')]
        elif os.path.isfile(self.path):
            if not self.path.endswith('.csv'):
                raise ValueError(f"The specified file '{self.path}' is not a CSV file.")
            self.files = [self.path]
        else:
            raise ValueError(f"The specified path '{self.path}' is neither a file nor a directory.")
        
    def get_dataframe(self):
        """
        Reads all CSV files and concatenates them into a single DataFrame.
        Renames the column corresponding to `note_text_column` to 'note_text'.
        
        Returns:
            pd.DataFrame: Concatenated DataFrame with the note text column renamed.
        """
        dataframes = []
        for file in self.files:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)
            
            # Ensure the note_text_column exists in the DataFrame
            if self.note_text_column not in df.columns:
                raise KeyError(f"The column '{self.note_text_column}' is not present in the file: {file}")
            
            # Rename the note_text_column to 'note_text'
            df = df.rename(columns={self.note_text_column: 'note_text'})
            
            # Add the DataFrame to the list
            dataframes.append(df)
        
        # Concatenate all DataFrames if there are multiple
        if dataframes:
            concatenated_df = pd.concat(dataframes, ignore_index=True)
        else:
            concatenated_df = pd.DataFrame()
        
        return concatenated_df