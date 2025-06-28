import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
import os
from tqdm import tqdm
import numpy as np

model_path = "./checkpoints/final_model"
config_path = "config.yaml"

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

base_model = AutoModelForCausalLM.from_pretrained(
    config['model']['name'],
    torch_dtype=getattr(torch, config['model']['torch_dtype']),
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

def translate(text, model, tokenizer, max_new_tokens=150):
    prompt = f"Translate the following text from Spanish to Galician.\nSpanish: {text}\nGalician:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,              # solo beam search
            do_sample=False,          # sin muestreo
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Galician:" in translation:
        translation = translation.split("Galician:")[-1].strip()
    
    return translation

dataset = load_dataset(
    config['data']['dataset_name'],
    f"{config['data']['source_lang']}-{config['data']['target_lang']}"
)

test_data = []
for item in dataset["devtest"]:
    if f"sentence_{config['data']['source_lang']}" in item:
        spanish = item[f"sentence_{config['data']['source_lang']}"]
        galician = item[f"sentence_{config['data']['target_lang']}"]
    else:
        spanish = item['sentence'][config['data']['source_lang']]
        galician = item['sentence'][config['data']['target_lang']]
    test_data.append({'spanish': spanish, 'galician': galician})

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

num_samples = 100
indices = np.random.choice(len(test_data), num_samples, replace=False)

predictions, references = [], []

for i in tqdm(indices):
    sample = test_data[i]
    pred = translate(sample['spanish'], model, tokenizer)
    predictions.append(pred)
    references.append([sample['galician']])

# Calcular métricas
bleu_score = bleu.compute(predictions=predictions, references=references)
chrf_score = chrf.compute(predictions=predictions, references=references)

print(f"BLEU: {bleu_score['score']:.2f}")
print(f"chrF: {chrf_score['score']:.2f}")

for i in range(5):
    print(f"\nEspañol: {test_data[indices[i]]['spanish']}")
    print(f"Referencia: {references[i][0]}")
    print(f"Predicción: {predictions[i]}")
