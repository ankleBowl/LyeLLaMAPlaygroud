import os
import requests
import sys
from llama_cpp import Llama

llm = None

model_type = "llama.cpp"

def load(**kwargs):
    global llm
    
    url = kwargs.get("url", None)
    path = kwargs.get("path", "")
    
    if url is not None:
        file_name = url.split("/")[-1]
        path = "models/" + file_name.split(".")[0] + "/" + file_name
        lora_path = "models/" + file_name.split(".")[0] + "/loras/"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            with open(path, "wb") as f:
                print("Downloading %s" % file_name)
                response = requests.get(url, stream=True)
                total_length = response.headers.get('content-length')

                if total_length is None: # no content length header
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                        sys.stdout.flush()
                        
    n_gpu_layers = kwargs.get("n_gpu_layers", 0)
    n_ctx = kwargs.get("n_ctx", 512)
        
    llm = Llama(model_path=path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    
    
    
def generate(prompt, max_length, temperature, **kwargs):
    global llm
    print(prompt)
    
    for output in llm(prompt, max_tokens=max_length, echo=True, stream=True, temperature=temperature):
        output = output["choices"][0]["text"]
        yield output
    
def unload():
    global llm
    del llm

def count_tokens(prompt):
    return len(llm.tokenize(prompt))

def get_loras():
    return []

def load_lora(lora_id):
    pass