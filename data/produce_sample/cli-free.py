import json
import time
import logging
import requests

from tqdm import tqdm
from multiprocessing import Pool

# this package is only valid in our lab's private internet 
from gpt import GPT

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_file = './cli_free.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# set your own pace, name of output file and url
NUM_OF_SESSIONS = 20
NUM_OF_PROCESSES = 30
output_file = 'sd-8125.json'
URL = "http://127.0.0.1:8125/generate"

role_of_chatGPT = "You are an artificial intelligence assistant."

def chat(num_of_processes):
    logger.info(f'Using {num_of_processes} processes.')

    for i in tqdm(range(NUM_OF_SESSIONS)):
        url = URL
        data = {
            "history": None, 
            "temperature": 0.7, 
            "max_new_tokens": 512
        }
        response = requests.post(url, json=data)

        gpt = GPT(user_name='kongchuyi', new_version='0.1.0')

        num_of_rounds = 0
        one_session_dict = {} 
        one_session_list = []

        while True:
            try:
                server_history = eval(response.text).get("history")
            except NameError:
                break
            q =  server_history[-1][-1]
            if "</s>" in q: 
                q = q.replace("</s>", "")

            answer = None
            err = 1
            while err or (not answer) or (answer == 'APIKey Error') or type(answer)!= str or ('context_length_exceeded' in answer) or ('invalid_api_key' in answer) or ('account_deactivated' in answer) or ('rate_limit_exceeded' in answer) or ('insufficient_quota' in answer) or ('获取请求参数user_name和parameters失败' in answer) or ('请求官方API失败。Error code:200') in answer:
                try:
                    _, answer = gpt.call(q, role_of_chatGPT)
                    err = 0
                except BaseException as e:  
                    print(f'\nThe {i+1}-session: \n {type(e)} \n {e}')

            num_of_rounds += 1
            one_session_list.append({"from": "human", "value": q})
            one_session_list.append({"from": "gpt", "value": answer})
                
            ans_list = ['Assistant']
            ans_list.append(answer)
            server_history.append(ans_list)
            data['history'] = server_history
            response = requests.post(url, json=data)
            
        one_session_dict['session_id'] = i+1
        one_session_dict['num_of_rounds'] = num_of_rounds
        one_session_dict['conversations'] = one_session_list

        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(one_session_dict, f, ensure_ascii=False)
            f.write('\n')
        
if __name__ == '__main__':
    start_time = time.time()

    with Pool() as pool:
        pool.map(chat, range(NUM_OF_PROCESSES))
    
    end_time = time.time()
    print(f'Elapsted {end_time-start_time} seconds.')
