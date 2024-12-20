import os
import json
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import Dataset

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

class Inference:
    def __init__(self, model_path, temperature=0, max_tokens=100):
        """
        Initialize the Inference class.

        Args:
            model_path (str): Path or identifier for the model to load.
            temperature (float, optional): Sampling temperature. Defaults to 0.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Instantiate the LLM and SamplingParams
        self.llm = LLM(model=self.model_path)
        self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)

    def generate(self, df, retries=3):
        """
        Given a DataFrame `df`, generates predictions by sending prompts to the LLM.

        Args:
            df (pd.DataFrame): Input DataFrame.
            retries (int, optional): Number of attempts to retry failed generations. Defaults to 3.

        Returns:
            pd.DataFrame: The input DataFrame with generated annotations.
        """
        # Load schema and prompt  
        # Construct the path to the JSON file relative to the current file
        schema_path = os.path.join(current_dir, "schema.json")
        with open(schema_path, "r") as f:
            schema = json.load(f)
        prompt_path = os.path.join(current_dir, "prompt.txt")
        with open(prompt_path, "r") as file:
            json_prompt = file.read()

        def formatting_prompts_func(examples, schema_str):
            inputs = examples["note"]
            conversations = []
            for input_text in inputs:
                prompt = {"content": json_prompt.format(input_text, schema_str), "role": "user"}
                conversations.append([prompt])
            return {"conversations": conversations}

        # Prepare the dataset
        df = df.fillna("")
        dataset = Dataset.from_pandas(df[["note"]])
        schema_str = json.dumps(schema, indent=4)
        dataset = dataset.map(
            lambda examples: formatting_prompts_func(examples, schema_str),
            batched=True
        )

        prompts = [[conv[0]] for conv in dataset['conversations']]

        results = [None] * len(prompts)
        generation_status = [None] * len(prompts)
        remaining_prompts = prompts.copy()
        remaining_indices = list(range(len(prompts)))

        for attempt in range(retries):
            print(f"Attempt {attempt + 1}/{retries}")
            
            # Process remaining prompts
            outputs = self.llm.chat(
                messages=remaining_prompts,
                sampling_params=self.sampling_params,
                use_tqdm=True,
            )

            failed_prompts = []
            failed_indices = []

            # Iterate over the model outputs
            for i, (index, output) in enumerate(zip(remaining_indices, outputs)):
                generated_text = output.outputs[0].text.strip().encode('unicode_escape').decode('utf-8')
                if not generated_text.endswith("}"):
                    generated_text += "}"

                try:
                    # Parse the text as JSON
                    parsed_json = json.loads(generated_text)

                    # Extract the required keys
                    result_row = {
                        "RCH_start_pred": parsed_json.get("HPI_Interval_Hx_start", None),
                        "RCH_end_pred": parsed_json.get("HPI_Interval_Hx_end", None),
                        "AP_start_pred": parsed_json.get("A&P_start", None),
                        "AP_end_pred": parsed_json.get("A&P_end", None),
                        "generation": True
                    }
                    results[index] = result_row
                    generation_status[index] = True

                except json.JSONDecodeError:
                    # If parsing fails, add to retry queue
                    failed_prompts.append(remaining_prompts[i])
                    failed_indices.append(index)

            # Update remaining prompts and indices for retry
            remaining_prompts = failed_prompts
            remaining_indices = failed_indices

            # Exit loop if all prompts are successfully parsed
            if not remaining_prompts:
                break

        # Handle any remaining failed prompts after final retry
        for index in remaining_indices:
            result_row = {
                "RCH_start_pred": None,
                "RCH_end_pred": None,
                "AP_start_pred": None,
                "AP_end_pred": None,
                "generation": False
            }
            results[index] = result_row
            generation_status[index] = False

        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)
        # Concatenate the new columns with the original DataFrame
        df = pd.concat([df, results_df], axis=1)
        return df