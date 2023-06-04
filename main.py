from flask import Flask, render_template
from flask_socketio import SocketIO
import os
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists("models"):
    os.mkdir("models")

model_types_to_methods = {}

for file in os.listdir("llm_support"):
    if file.endswith(".py") and not file.startswith("."):
        print("Loading " + file)
        exec("import llm_support." + file[:-3] + " as " + file[:-3])
        model_types_to_methods[file[:-3]] = [
            eval(file[:-3] + ".load"), 
            eval(file[:-3] + ".generate"),
            eval(file[:-3] + ".unload"), 
            eval(file[:-3] + ".count_tokens"), 
            eval(file[:-3] + ".get_loras")
            eval(file[:-3] + ".load_lora")
        ]
        print(model_types_to_methods[file[:-3]])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)

current_model = {}

temperature = 1.0
max_length = 64

data = None
with open("config.json", "r") as f:
    data = json.load(f)

@app.route('/')
def index():
    model_template = """<div style="background: linear-gradient(90deg, {HEX} 0%, rgba(0,0,0,0) 10%);" onclick="loadModel('{MODEL_ID}')" class="model">{NAME_NO_AUTHOR}</div>"""
    index = render_template('index.html')
    html = ""
    for model in data["models"]:
        color = data["types"][data["models"][model]["type"]]["color"]
        html += model_template.format(HEX=color, MODEL_ID=model, NAME_NO_AUTHOR=model)
    index = index.replace("$models", html)
    return index

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

is_counting_tokens = False

@socketio.on('message')
def handle_message(message):
    global current_model
    
    global max_length
    global temperature
        
    print('Received message: ' + message)
    # socketio.emit('message', message, broadcast=True)
    msg_type = message.split(":", 1)[0]
    if msg_type == "load_model":
        global current_model
        
        model_id = message.split(":")[1]
        if model_id != current_model:
            try:
                socketio.emit('message', "block_screen:Loading model...")
                print("Loading model...")
                
                if "type" in current_model:
                    model_types_to_methods[current_model["type"]][2]()
                
                current_model = data["models"][model_id]
                model_types_to_methods[current_model["type"]][0](**current_model["instantiate_params"])
        
                socketio.emit('message', "unblock_screen")
                print("Loaded model")
            except Exception as e:
                socketio.emit('message', "error:" + str(e))
    if msg_type == "complete":
        try:
            prompt = message.split(":", 1)[1]
            
            if "generate_params" not in current_model:
                current_model["generate_params"] = {}
                
            for generated in model_types_to_methods[current_model["type"]][1](prompt, max_length, temperature, **current_model["generate_params"]):
                socketio.emit('message', "completion:" + generated)
                
            socketio.emit('message', "finish_completion")
        except Exception as e:
            socketio.emit('message', "error:" + str(e))
            socketio.emit('message', "finish_completion")
    if msg_type == "count_tokens":
        global is_counting_tokens
        if is_counting_tokens:
            return
        is_counting_tokens = True
        prompt = message.split(":", 1)[1]
        tokens = model_types_to_methods[current_model["type"]][3](prompt)
        socketio.emit('message', "token_count:" + str(tokens))
        is_counting_tokens = False
    if msg_type == "set_var":
        split_at_all = message.split(":", 2)
        var_name = split_at_all[1]
        var_value = split_at_all[2]
        if var_name == "temperature":
            temperature = float(var_value)
        if var_name == "max_length":
            max_length = int(var_value)
            print("Max length set to " + str(max_length))
    if msg_type == "get_loras":
        loras = model_types_to_methods[current_model["type"]][4]()
        loras.append("None")
        string_rep = ""
        for lora in loras:
            string_rep += lora + ";"
        string_rep = string_rep[:-1]
        socketio.emit('message', "loras:" + string_rep)
    if msg_type == "load_lora":
        socketio.emit('message', "block_screen:Loading LoRA...")
        print("Loading lora")
        lora = message.split(":", 1)[1]
        if lora == "None":
            model_types_to_methods[current_model["type"]][2]()
            model_types_to_methods[current_model["type"]][0](**current_model["instantiate_params"])
        else:
            model_types_to_methods[current_model["type"]][5](lora)
        socketio.emit('message', "unblock_screen")
if __name__ == '__main__':
    socketio.run(app, port=7001)
