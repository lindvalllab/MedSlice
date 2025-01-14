from datasets import Dataset
import json

with open("../../schema.json", "r") as f:
    schema = json.load(f)
with open("../../prompt.txt", "r") as file:
    json_prompt = file.read()

def formatting_prompts_func(examples, schema_str):
    inputs = examples["note_text"]
    responses = [
        {"HPI_Interval_Hx_start": hpi_start, "HPI_Interval_Hx_end": hpi_end, "A&P_start": ap_start, "A&P_end": ap_end}
        for hpi_start, hpi_end, ap_start, ap_end in zip(
            examples["HPI_Interval_Hx_start_str_gt"], 
            examples["HPI_Interval_Hx_end_str_gt"], 
            examples["A&P_start_str_gt"], 
            examples["A&P_end_str_gt"]
        )
    ]
    conversations = []
    for input_text, response_json in zip(inputs, responses):
        response_str = json.dumps(response_json)
        prompt = {"content": json_prompt.format(input_text, schema_str), "role": "user"}
        answer = {"content": response_str, "role": "assistant"}
        conversations.append([prompt, answer])
    
    return {"conversations": conversations}

def formatting_prompts_func_2(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def generate_prompts(df, tokenizer):
    df_selected = df[["note_text", "HPI_Interval_Hx_start_str_gt", "HPI_Interval_Hx_end_str_gt", "A&P_start_str_gt", "A&P_end_str_gt"]].copy()
    df_selected = df_selected.fillna("")
    dataset_gen = Dataset.from_pandas(df_selected)
    schema_str = json.dumps(schema, indent=4)
    dataset_gen = dataset_gen.map(
        lambda examples: formatting_prompts_func(examples, schema_str),
        batched=True
    )
    dataset = dataset_gen.map(
        lambda examples: formatting_prompts_func_2(examples, tokenizer),
        batched=True
    )
    dataset = dataset.remove_columns(["note_text", "HPI_Interval_Hx_start_str_gt", "HPI_Interval_Hx_end_str_gt", "A&P_start_str_gt", "A&P_end_str_gt"])
    return dataset