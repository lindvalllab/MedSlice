import argparse
import os
from src.preprocessing import Preprocessing
from src.inference import Inference, get_pred_indexes
from src.report import PDFGenerator, postprocessing


def main(model_path, data_path, sectioned_output_path, pdf_output_path=None, note_text_column=None):
    # Step 1: Preprocess the data
    print(f"Preprocessing data from {data_path}...")
    preprocessor = Preprocessing(data_path, note_text_column=note_text_column)
    df = preprocessor.get_processed_dataframe()

    # Step 2: Perform inference
    print(f"Loading model from {model_path}...")
    inference_engine = Inference(model_path)
    print("Generating predictions...")
    llm_output = inference_engine.generate(df)
    llm_output = get_pred_indexes(llm_output)

    # Step 3: Extracting the sections using LLM outputs
    print("Extracting the sections using LLM outputs...")
    postprocessed_df = postprocessing(llm_output)
    print(f"Saving sectioned data to {sectioned_output_path}...")
    postprocessed_df.to_csv(sectioned_output_path, index=False)

    # Step 4: Generate PDF (if output path is provided)
    if pdf_output_path:
        print(f"Generating PDF at {pdf_output_path}...")
        generator = PDFGenerator(
            llm_output,
            'RCH_start_pred',
            'RCH_end_pred',
            'AP_start_pred',
            'AP_end_pred'
        )
        generator.convert_to_pdf(pdf_output_path)

    print("Script execution completed.")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run inference on the given dataset using a specified model.")

    # Add arguments
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("data_path", type=str, help="Path to the input data.")
    parser.add_argument("sectioned_output_path", type=str, help="Path to save the sectioned data.")
    parser.add_argument("--pdf_output_path", type=str, default=None, help="Path to save the generated PDF.")
    parser.add_argument("--note_text_column", type=str, default=None, help="Column containing the notes.")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(
        model_path=args.model_path,
        data_path=args.data_path,
        sectioned_output_path=args.sectioned_output_path,
        pdf_output_path=args.pdf_output_path,
        note_text_column=args.note_text_column
    )