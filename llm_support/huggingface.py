from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel 
import torch
import os

current_model = None
current_tokenizer = None
current_model_name = None

model_type = "huggingface"

def load(**kwargs):
    global current_model
    global current_tokenizer
    global current_model_name
    
    huggingface_id = kwargs.get("model", None)
    tokenizer_id = kwargs.get("tokenizer", None)
    
    current_model_name = huggingface_id.replace("/", "_")
    
    model_folder = "models/" + current_model_name + "/loras"
    os.makedirs(model_folder, exist_ok=True)
    
    if tokenizer_id is None:
        tokenizer_id = huggingface_id
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    if not kwargs.get("four_bit", True):
        bnb_config = None
    current_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    current_tokenizer.pad_token = current_tokenizer.eos_token
    
    device_map = {"":0}
    if kwargs.get("use_cpu", False):
        device_map = {"": "cpu"}
    
    current_model = AutoModelForCausalLM.from_pretrained(huggingface_id, quantization_config=bnb_config, device_map=device_map,trust_remote_code=True)
    
def generate(prompt, max_length, temperature, **kwargs):
    global current_model
    global current_tokenizer
    
    inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)
    if not kwargs.get("use_token_type_ids", True):
        del inputs["token_type_ids"]
    outputs = current_model.generate(**inputs, max_new_tokens=max_length)
    yield current_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")

def unload():
    global current_model
    global current_tokenizer
    current_model = None
    current_tokenizer = None
    
def count_tokens(prompt):
    inputs = current_tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"].shape[1]

def get_loras():
    lora_path = "models/" + current_model_name + "/loras"
    loras = []
    for file in os.listdir(lora_path):
        if os.path.isdir(os.path.join(lora_path, file)):
            loras.append(file)
    return loras

def load_lora(lora_id):
    global current_model
    lora_path = "models/" + current_model_name + "/loras/" + lora_id
    current_model = PeftModel.from_pretrained(current_model, lora_path)