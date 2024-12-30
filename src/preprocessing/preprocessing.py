import os
import json
import pandas as pd

class Preprocessing:
    def __init__(self, path, note_text_column=None):
        """
        Initialize the Preprocessing class.
        
        Args:
            path (str): Path to a CSV/JSON file or a folder containing CSV/JSON files.
            note_text_column (str, optional): Name of the column corresponding to note text.
                                              If not provided, defaults to:
                                                - "note_text" if non-annotated
                                                - Extract "notes" from the JSON in 'data' column if annotated.
        """
        self.path = path
        self.note_text_column = note_text_column
        
        # Validate the path
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The specified path '{self.path}' does not exist.")
        
        # Determine files
        if os.path.isdir(self.path):
            # Include both CSV and JSON files
            self.files = [
                os.path.join(self.path, f) for f in os.listdir(self.path) 
                if f.lower().endswith('.csv') or f.lower().endswith('.json')
            ]
        elif os.path.isfile(self.path):
            if not (self.path.lower().endswith('.csv') or self.path.lower().endswith('.json')):
                raise ValueError(f"The specified file '{self.path}' is not a CSV or JSON file.")
            self.files = [self.path]
        else:
            raise ValueError(f"The specified path '{self.path}' is neither a file nor a directory.")
        
    def get_dataframe(self):
        """
        Reads all CSV/JSON files and concatenates them into a single DataFrame.
        
        Returns:
            pd.DataFrame: Concatenated DataFrame.
        """
        dataframes = []
        for file in self.files:
            if file.lower().endswith('.csv'):
                df = pd.read_csv(file)
            elif file.lower().endswith('.json'):
                df = pd.read_json(file, orient='records')
            else:
                # This should not happen due to checks in __init__
                continue
            
            dataframes.append(df)
        
        if dataframes:
            concatenated_df = pd.concat(dataframes, ignore_index=True)
        else:
            concatenated_df = pd.DataFrame()
        
        return concatenated_df

    @staticmethod
    def process_annotations(json_string):
        """
        Process a JSON string of annotations and return the start/end for 'HPI_Interval_Hx' and 'A&P'.
        
        Args:
            json_string: A JSON-formatted string containing annotations data (list of annotation dicts).
        
        Returns:
            Tuple: (RCH_start_gt, RCH_end_gt, AP_start_gt, AP_end_gt)
        """
        RCH_start_gt = pd.NA
        RCH_end_gt = pd.NA
        AP_start_gt = pd.NA
        AP_end_gt = pd.NA
        if isinstance(json_string, list):
            json_string = json_string[0]
        if isinstance(json_string, str):
            try:
                parsed_value = json.loads(json_string)
                for item in parsed_value:
                    labels = item.get("labels", [])
                    start = item.get("start")
                    end = item.get("end")
                    
                    for label in labels:
                        if label == "HPI_Interval_Hx" and pd.isna(RCH_start_gt):
                            RCH_start_gt = start
                            RCH_end_gt = end
                        elif label == "A&P" and pd.isna(AP_start_gt):
                            AP_start_gt = start
                            AP_end_gt = end
            except (json.JSONDecodeError, TypeError):
                return RCH_start_gt, RCH_end_gt, AP_start_gt, AP_end_gt
        elif isinstance(json_string, dict):
            results = json_string.get("result", [])
            for result in results:
                value = result.get("value", {})
                labels = value.get("labels", [])
                if labels == ["Interval History"] and pd.isna(RCH_start_gt):
                    RCH_start_gt = value.get("start")
                    RCH_end_gt = value.get("end")
                if labels == ["Assessment and Plan"] and pd.isna(AP_start_gt):
                    AP_start_gt = value.get("start")
                    AP_end_gt = value.get("end")
        
        return RCH_start_gt, RCH_end_gt, AP_start_gt, AP_end_gt

    @staticmethod
    def extract_notes_from_data(data_string):
        """
        Extract 'notes' from a JSON string stored in the 'data' column.
        
        Args:
            data_string: A JSON-formatted string containing at least a 'notes' key.
        
        Returns:
            The value of 'notes' or pd.NA if it doesn't exist.
        """
        if not isinstance(data_string, dict):
            return pd.NA
        return data_string.get("notes", pd.NA)

    def transform_annotated_df(self, df, annotations_column='annotations'):
        """
        For annotated DataFrames:
        - Parse each row's annotations JSON and extract RCH_start_gt, RCH_end_gt, AP_start_gt, AP_end_gt.
        - If note_text_column is not provided, also extract 'notes' from the 'data' column JSON and create 'note' column.
        
        The number of rows remains the same as the original DataFrame.
        
        Args:
            df (pd.DataFrame): The annotated DataFrame.
            annotations_column (str): The name of the column containing the annotations JSON.
        
        Returns:
            pd.DataFrame: The original DataFrame plus the new columns.
        """
        rch_starts = []
        rch_ends = []
        ap_starts = []
        ap_ends = []
        
        for _, row in df.iterrows():
            RCH_start_gt, RCH_end_gt, AP_start_gt, AP_end_gt = self.process_annotations(row.get(annotations_column, None))
            rch_starts.append(RCH_start_gt)
            rch_ends.append(RCH_end_gt)
            ap_starts.append(AP_start_gt)
            ap_ends.append(AP_end_gt)
        
        df['RCH_start_gt'] = rch_starts
        df['RCH_end_gt'] = rch_ends
        df['AP_start_gt'] = ap_starts
        df['AP_end_gt'] = ap_ends

        # If note_text_column is not provided for annotated data, we must extract notes from the JSON in 'data'.
        if self.note_text_column is None:
            if 'data' not in df.columns:
                raise KeyError("The 'data' column is not present in the annotated data. Cannot extract 'notes'.")
            
            notes_column = df['data'].apply(self.extract_notes_from_data)
            df['note'] = notes_column
        
        return df

    def get_processed_dataframe(self):
        """
        Load and process the input data based on whether annotations are present.
        
        If there's an 'annotations' column (annotated format):
          - If note_text_column is provided, ensure it exists and rename it to 'note'.
          - If note_text_column is not provided, extract 'notes' from the 'data' JSON and create 'note' column.
          - Add RCH_start_gt, RCH_end_gt, AP_start_gt, AP_end_gt columns by calling transform_annotated_df.
        
        If there's no 'annotations' column (non-annotated format):
          - If note_text_column is provided, use it, ensuring it exists, and rename it to 'note'.
          - Otherwise, default to "note_text", ensure it exists, rename it to 'note'.
        
        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        df = self.get_dataframe()
        
        # Determine if annotated
        annotations_column = None
        if 'annotations' in df.columns:
            annotations_column = 'annotations'
        elif 'label' in df.columns:
            annotations_column = 'label'
        
        # Annotation extraction as ground truth
        if annotations_column:
            df = self.transform_annotated_df(df, annotations_column=annotations_column)
        
        if "note" not in df.columns:
            # Renaming the note column for consistency in downstream processing
            if self.note_text_column is not None:
                col_to_rename = self.note_text_column
            else:
                # Default to "note_text" if not provided
                col_to_rename = 'note_text'
            if col_to_rename not in df.columns:
                raise KeyError(f"The column '{col_to_rename}' is not present in the data.")
            df = df.rename(columns={col_to_rename: 'note'})
        
        return df