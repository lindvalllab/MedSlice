# Sectioning

## Overview
This repository serves to generate RCH (History of Present Illness and Interval History) and AP (Assessment and Plan) sections for progress notes. The code provides tools for preprocessing data, running model inference, and generating output reports in CSV format, with the option to output a PDF to visualize the sections. You can use the `sectioning.py` script to execute these tasks directly from the command line. Follow the steps below to set up the required environment and learn how to use the script.

## Getting started

1. **Clone Repository and access it**
   ```bash
   git clone https://github.com/lindvalllab/sectioning.git
   cd sectioning

2. **Install Conda (if not already installed)**  
   If you don't have Conda installed, you can download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

3. **Create the Conda Environment and activate it**  
   Use the provided `environment.yml` file to create the environment. Run the following command:
   ```bash
   conda env create -f environment.yml
   conda activate sectioning

4. **Install VLLM and Unsloth**
   You will need to install VLLM and Unsloth with pip:
   ```bash
   pip install vllm
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps trl peft accelerate bitsandbytes

## Usage

The entry points for running the workflow are the `sectioning.py` and `finetuning.py` scripts. Each script takes specific arguments and can be run directly from the command line.

### 1. `finetuning.py`

This script fine-tunes a model on custom data using rsLoRA.

#### Arguments:
* `model_name`: Name of the pre-trained model. **Required**.
* `data_path`: Path to the dataset CSV file. **Required**.
* `--n_epochs`: Number of training epochs. Optional, defaults to `5`.
* `--r_lora`: LoRA rank. Optional, defaults to `16`.
* `--use_rslora`: Whether to use rsLoRA. Optional, defaults to `True`.
* `--output_folder`: Folder to save the fine-tuned model. Optional, defaults to `"models"`.
* `--max_seq_length`: Maximum sequence length. Optional, defaults to `8192`.
* `--load_in_4bit`: Whether to load the model in 4-bit precision. Optional, defaults to `False`.

#### Example:
```bash
python finetuning.py "unsloth/Llama-3.2-1B-Instruct" data/path/to/dataset.csv --n_epochs 5 --r_lora 16
```

### 2. `sectioning.py`

This script runs the sectioning workflow for extracting RCH and AP sections from progress notes.

#### Arguments:
* `model_path`: Path to the trained model directory. **Required**.
* `data_path`: Path to the input data file. **Required**.
* `sectioned_output_path`: Path to save the postprocessed CSV data. **Required**.
* `--pdf_output_path`, `-p`: Path to save the generated PDF report. If not provided, no PDF will be generated.
* `--note_text_column`: Column containing the notes. Optional, defaults to `None`.

#### Example:
```bash
python sectioning.py /path/to/model /path/to/data.csv /path/to/output.csv --pdf_output_path /path/to/report.pdf --note_text_column "notes"
```

## Project Organization

      ├── LICENSE                <- GPL-3.0 License
      ├── README.md              <- The top-level README for developers using this project.
      ├── data                   <- A placeholder for your data, one or several csv files.
      ├── environment.yml        <- The requirements file for reproducing the sectioning environment.
      ├── models                 <- A placeholder for your models, has to be readable by VLLM.
      ├── sectioning.py          <- Main script to run the sectioning. 
      └── src                    <- Additional source code for use in this project.
         ├── __init__.py               <- Makes src a Python module.
         ├── benchmarking              <- Scripts to benchmark the sectioning tool, when the ground truth is provided.
         │   ├── __init__.py                 <- Makes benchmarking a Python module.
         │   └── scorer.py                   <- Code for the sectioning scorer.
         ├── inference                 <- Scripts to perform inference using VLLM and fuzzy matching.
         │   ├── __init__.py                 <- Makes inference a Python module.
         │   ├── inference.py                <- Code for VLLM inference.
         │   ├── output_matching.py          <- Code for fuzzy matching between LLM outputs and input.
         │   ├── prompt.txt                  <- Prompt passed to the model, as a txt file.
         │   └── schema.json                 <- Output schema passed to the model, as a json file.
         ├── preprocessing             <- Scripts to preprocess the inputs before downstream processing. You can adapt this to your input format and structure.
         │   ├── __init__.py                 <- Makes preprocessing a Python module.
         │   └── preprocessing.py            <- Code for preprocessing the data.
         └── report                    <- Scripts to report the sections as pdf and csv file.
            ├── __init__.py                  <- Makes report a Python module.
            ├── pdfgenerator.py              <- Code to generate the PDF file with overlayed LLM sections.
            └── postprocessing.py            <- Code to extract the sections as text using indexes found with fuzzy matching.
