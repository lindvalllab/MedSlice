import tempfile
import pandas as pd
from weasyprint import HTML

class PDFGenerator:
    """
    A class to generate an HTML page with predicted highlighted sections for each note.
    This version:
    - Has two possible predicted annotations: RCH (yellow) and A&P (blue).
    - Has two optional sets of ground truth annotations: RCH (red text) and A&P (purple text).
    - If both ground truths apply to the same segment, both classes are applied, and purple overrides red.
    - Ensures that a label for a given predicted annotation appears only once per annotation segment.
    """

    def __init__(self, notes_df, 
                 rch_start_col, rch_end_col, ap_start_col, ap_end_col, 
                 gt_rch_start_col=None, gt_rch_end_col=None,
                 gt_ap_start_col=None, gt_ap_end_col=None):
        """
        Initialize with:
        - notes_df: DataFrame with 'note_text' and prediction columns.
        - rch_start_col, rch_end_col: column names for RCH annotation (prediction)
        - ap_start_col, ap_end_col: column names for A&P annotation (prediction)
        - gt_rch_start_col, gt_rch_end_col: optional column names for RCH ground truth
        - gt_ap_start_col, gt_ap_end_col: optional column names for A&P ground truth
        """
        self.notes_df = notes_df.sort_index()
        self.rch_start_col = rch_start_col
        self.rch_end_col = rch_end_col
        self.ap_start_col = ap_start_col
        self.ap_end_col = ap_end_col
        self.gt_rch_start_col = gt_rch_start_col
        self.gt_rch_end_col = gt_rch_end_col
        self.gt_ap_start_col = gt_ap_start_col
        self.gt_ap_end_col = gt_ap_end_col
        self.html_content = ""

    def create_html_content(self):
        """
        Create the HTML content with notes and highlighted predicted sections (RCH, A&P),
        and optionally ground truth in red or purple.
        """
        # Begin the HTML content
        self.html_content += "<html><head><style>"
        self.html_content += "body { font-family: Arial, sans-serif; line-height: 1.5; margin: 20px; }"
        self.html_content += "p { text-align: justify; }"
        # Prediction highlights
        self.html_content += ".highlight-first { background-color: yellow; }"   # RCH prediction
        self.html_content += ".highlight-second { background-color: lightblue; }" # A&P prediction
        # Ground truth text colors
        self.html_content += ".ground-truth-rch { color: red; }"     # RCH GT
        self.html_content += ".ground-truth-ap { color: purple; }"   # A&P GT

        self.html_content += (
            ".label-type {"
            "display: inline-block; font-size: 9px; font-weight: bold;"
            "background-color: #E0E0E0; border: 1px solid black; padding: 2px 4px;"
            "border-radius: 3px; text-align: center; margin-right: 5px;"
            "vertical-align: middle; margin-left: 5px}"
        )
        self.html_content += ".page-break { page-break-before: always; }"
        self.html_content += ".legend { text-align: center; margin-bottom: 10px; }"
        self.html_content += "</style></head><body>"

        for index, row in self.notes_df.iterrows():
            # Add a page break before each new note (except the first one)
            if index > 0:
                self.html_content += "<div class='page-break'></div>"

            # Add legend for each page
            self.html_content += self.create_legend()

            self.html_content += f"<h2>Note {index}</h2>"

            note_text = row['note_text']
            # Generate annotations from the given columns
            pred_annotations, gt_rch_annotation, gt_ap_annotation = self.get_annotations_from_row(row)
            highlighted_text = self.highlight_text(note_text, pred_annotations, gt_rch_annotation, gt_ap_annotation)

            self.html_content += f"<p>{highlighted_text}</p>"

        self.html_content += "</body></html>"

    def create_legend(self):
        """
        Create a centered legend line that shows:
        - RCH ground truth (red text) if provided
        - A&P ground truth (purple text) if provided
        """
        legend_html = "<div class='legend'>"

        # Only show RCH GT if columns are provided
        if self.gt_rch_start_col is not None and self.gt_rch_end_col is not None:
            legend_html += "<span class='ground-truth-rch'>RCH GT</span> &nbsp;"

        # Only show A&P GT if columns are provided
        if self.gt_ap_start_col is not None and self.gt_ap_end_col is not None:
            legend_html += "<span class='ground-truth-ap'>A&P GT</span>"

        legend_html += "</div>"
        return legend_html

    def get_annotations_from_row(self, row):
        """
        Returns a tuple: (pred_annotations, gt_rch_annotation, gt_ap_annotation)
        """
        pred_annotations = []

        def valid_interval(s, e):
            return not pd.isna(s) and not pd.isna(e)

        # Handle RCH annotation
        rch_start = row[self.rch_start_col]
        rch_end = row[self.rch_end_col]
        if valid_interval(rch_start, rch_end):
            pred_annotations.append({"start": int(rch_start), "end": int(rch_end), "labels": ["RCH"]})

        # Handle A&P annotation
        ap_start = row[self.ap_start_col]
        ap_end = row[self.ap_end_col]
        if valid_interval(ap_start, ap_end):
            pred_annotations.append({"start": int(ap_start), "end": int(ap_end), "labels": ["A&P"]})

        # Handle RCH GT
        gt_rch_annotation = None
        if self.gt_rch_start_col is not None and self.gt_rch_end_col is not None:
            gt_rch_start = row[self.gt_rch_start_col]
            gt_rch_end = row[self.gt_rch_end_col]
            if valid_interval(gt_rch_start, gt_rch_end):
                gt_rch_annotation = {"start": int(gt_rch_start), "end": int(gt_rch_end)}

        # Handle A&P GT
        gt_ap_annotation = None
        if self.gt_ap_start_col is not None and self.gt_ap_end_col is not None:
            gt_ap_start = row[self.gt_ap_start_col]
            gt_ap_end = row[self.gt_ap_end_col]
            if valid_interval(gt_ap_start, gt_ap_end):
                gt_ap_annotation = {"start": int(gt_ap_start), "end": int(gt_ap_end)}

        return pred_annotations, gt_rch_annotation, gt_ap_annotation

    def highlight_text(self, note_text, pred_annotations, gt_rch_annotation, gt_ap_annotation):
        """
        Highlight the sections of the note text based on predictions and ground truth.
        If segment is covered by both GT and predictions, apply both styles.
        
        Ensure that a label for a given predicted annotation is only displayed once.
        """
        boundaries = {0, len(note_text)}
        for ann in pred_annotations:
            boundaries.add(ann['start'])
            boundaries.add(ann['end'])

        if gt_rch_annotation:
            boundaries.add(gt_rch_annotation['start'])
            boundaries.add(gt_rch_annotation['end'])

        if gt_ap_annotation:
            boundaries.add(gt_ap_annotation['start'])
            boundaries.add(gt_ap_annotation['end'])

        sorted_boundaries = sorted(boundaries)
        
        intervals = []
        for i in range(len(sorted_boundaries) - 1):
            intervals.append((sorted_boundaries[i], sorted_boundaries[i+1]))

        highlighted_text = ""
        # Keep track of which annotations have had their label displayed
        displayed_annotations = set()
        
        for start, end in intervals:
            segment = note_text[start:end]

            # Check GT coverage
            rch_gt_covers = gt_rch_annotation and gt_rch_annotation['start'] <= start and gt_rch_annotation['end'] >= end
            ap_gt_covers = gt_ap_annotation and gt_ap_annotation['start'] <= start and gt_ap_annotation['end'] >= end

            # Check prediction coverage
            covering_pred = [ann for ann in pred_annotations if ann['start'] <= start and ann['end'] >= end]

            # Decide classes
            css_classes = []
            label_tag = ""

            if covering_pred:
                # Take the first covering prediction
                pred_ann = covering_pred[0]
                label_type = pred_ann['labels'][0] if pred_ann['labels'] else "Unknown"
                
                # Determine if we should display the label
                ann_id = (pred_ann['start'], pred_ann['end'], label_type)
                # Only add label if we haven't displayed it before
                if ann_id not in displayed_annotations:
                    label_tag = f"<span class='label-type'>{label_type}: </span>"
                    displayed_annotations.add(ann_id)

                # Determine background color by label
                if label_type == "RCH":
                    css_classes.append('highlight-first')   # Yellow
                elif label_type == "A&P":
                    css_classes.append('highlight-second')  # Blue

            # Ground truth text colors
            if rch_gt_covers:
                css_classes.append('ground-truth-rch')   # Red text
            if ap_gt_covers:
                css_classes.append('ground-truth-ap')    # Purple text

            if css_classes:
                class_str = " ".join(css_classes)
                highlighted_text += f"{label_tag}<span class='{class_str}'>{segment}</span>"
            else:
                highlighted_text += segment
        
        return highlighted_text

    def convert_to_pdf(self, output):
        """
        Convert the generated HTML content to a PDF using WeasyPrint.
        """
        with tempfile.NamedTemporaryFile(suffix=".html") as temp_html:
            self.create_html_content()
            temp_html.write(self.html_content.encode('utf-8'))
            temp_html_path = temp_html.name
            HTML(temp_html_path).write_pdf(output)