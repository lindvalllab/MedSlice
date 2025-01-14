import argparse
import pandas as pd
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from src.finetuning import generate_prompts

def main(model_name, data_path, n_epochs=5, r_lora=16, use_rslora=True, output_folder="models", max_seq_length=8192, load_in_4bit=False):
    dtype = None

    print(f"Loading model {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=r_lora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=r_lora,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=use_rslora,
        loftq_config=None,
    )

    chat_template = None
    if "Llama" in model_name:
        chat_template = "llama-3.1"
    elif "Mistral" in model_name:
        chat_template = "mistral"
    elif "Phi" in model_name:
        chat_template = "phi-3"
    elif "gemma" in model_name:
        chat_template = "gemma2"
    
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    
    print(f"Reading data from {data_path}...")
    df_finetuning = pd.read_csv(data_path)
    dataset = generate_prompts(df_finetuning, tokenizer)
    if "Mistral" in model_name:
        dataset = dataset.add_column("labels", ["foo"] * len(dataset))

    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer) if "Llama" in model_name else None,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=n_epochs,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            save_strategy="steps",
            save_steps=10,
        ),
    )

    if "Llama" in model_name:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    print("Training model...")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained_merged(output_folder, tokenizer, save_method="merged_16bit")
    print(f"Model saved to {output_folder}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a FastLanguageModel with LoRA using unsloth.")

    parser.add_argument("model_name", type=str, help="Name of the pre-trained model.")
    parser.add_argument("data_path", type=str, help="Path to the dataset.")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--r_lora", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--use_rslora", type=bool, default=True, help="Whether to use rsLoRA.")
    parser.add_argument("--output_folder", type=str, default="models", help="Folder to save the fine-tuned model.")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length.")
    parser.add_argument("--load_in_4bit", type=bool, default=False, help="Whether to load model in 4-bit precision.")

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        data_path=args.data_path,
        n_epochs=args.n_epochs,
        r_lora=args.r_lora,
        use_rslora=args.use_rslora,
        output_folder=args.output_folder,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )