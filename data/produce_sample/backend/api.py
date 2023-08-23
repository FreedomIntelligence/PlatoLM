import logging
from copy import deepcopy

from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks

from inference import load_model, produce_history

app = FastAPI()

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_file = './api.log'
file_handler = logging.FileHandler(log_file, mode='a')  
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

model = None
tokenizer = None
# set your model path
path = ''

async def load_model_background():
    global model, tokenizer
    logger.info(f'will load {path}..')
    model, tokenizer = await load_model(model_path = path, device = 'cuda', num_gpus = 1)

@app.on_event("startup")
async def startup_event(): 
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model_background)
    await background_tasks() 

class Request(BaseModel):
    conv_template: str = None 
    history: list = None
    temperature: float = 0.7 
    max_new_tokens: int = 512

@app.post("/generate")
def generate(req: Request):
    if not req.history:
        logger.info(f'###### Restart one session!!')
    req_history_copy = deepcopy(req.history) 
    history = produce_history(model, tokenizer, req_history_copy, model_path = path,
                              temperature = req.temperature, max_new_tokens = req.max_new_tokens)
    if not history:
        return {"hisotry": None}
    return {"history": history}
