import os
import torch
import yaml
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config):

    logger.info(f"Cargando modelo {config['model']['name']}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=getattr(torch, config['model']['torch_dtype']),
        device_map=config['model']['device_map'],
        trust_remote_code=True,
        use_cache=config['model']['use_cache']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if config['training'].get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
    
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memoria usada: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def prepare_datasets(tokenizer, config):
    logger.info("Cargando dataset FLORES-200...")
    
    try:
        dataset = load_dataset(
            config['data']['dataset_name'], 
            f"{config['data']['source_lang']}-{config['data']['target_lang']}"
        )
    except:
        dataset = load_dataset("facebook/flores", "all")
    
    def format_examples(examples):
        formatted_texts = []
        if f"sentence_{config['data']['source_lang']}" in examples:
            source_key = f"sentence_{config['data']['source_lang']}"
            target_key = f"sentence_{config['data']['target_lang']}"
        else:
            source_texts = [ex[config['data']['source_lang']] for ex in examples['sentence']]
            target_texts = [ex[config['data']['target_lang']] for ex in examples['sentence']]
            
            for src, tgt in zip(source_texts, target_texts):
                instruction = f"Translate the following text from Spanish to Galician.\nSpanish: {src}\nGalician: {tgt}"
                formatted_texts.append(instruction)
            
            return tokenizer(
                formatted_texts,
                max_length=config['data']['max_length'],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        for i in range(len(examples[source_key])):
            src = examples[source_key][i]
            tgt = examples[target_key][i]
            instruction = f"Translate the following text from Spanish to Galician.\nSpanish: {src}\nGalician: {tgt}"
            formatted_texts.append(instruction)
        
        model_inputs = tokenizer(
            formatted_texts,
            max_length=config['data']['max_length'],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    
    train_dataset = dataset["dev"].map(
        format_examples,
        batched=True,
        remove_columns=dataset["dev"].column_names
    )
    
    eval_dataset = dataset["devtest"].map(
        format_examples,
        batched=True,
        remove_columns=dataset["devtest"].column_names
    )
    
    logger.info(f"Train: {len(train_dataset)} ejemplos, Eval: {len(eval_dataset)} ejemplos")
    
    return train_dataset, eval_dataset


def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy="steps",  ###
        save_strategy="steps",  ###
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  
        report_to="none",
        dataloader_pin_memory=True,
        remove_unused_columns=False
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Iniciando entrenamiento...")
    trainer.train()
    
    logger.info("Guardando modelo final...")
    final_path = os.path.join(config['training']['output_dir'], "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info(f"Modelo guardado en {final_path}")
    
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_path


def main():
    config = load_config("config.yaml")
    
    model, tokenizer = setup_model_and_tokenizer(config)
    
    train_dataset, eval_dataset = prepare_datasets(tokenizer, config)
    
    final_model_path = train_model(model, tokenizer, train_dataset, eval_dataset, config)
    
    logger.info("Entrenamiento completado exitosamente!")
    return final_model_path


if __name__ == "__main__":
    main()