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
log_file = './cli-wizard.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

role_of_chatGPT = "You are an artificial intelligence assistant."

# set your own pace, name of files and url
NUM_OF_PROCESSES = 100
URL = "http://127.0.0.1:8125/generate"
output_path = 'ww-8125.json'
wizard_path = './wizard_sg_10k.json'
# special cases
exist_path = './ww-8996.json'
exceed_path = './ww-exc-id-201.json'

def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        dict_list = json.load(file)
    return dict_list

def chat(process_id, num_of_processes, total_list):
    logger.info(f'Using {num_of_processes} processes.')

    total_num_list = len(total_list)
    dicts_per_process = (total_num_list + num_of_processes - 1) // num_of_processes
    start_idx = process_id * dicts_per_process
    end_idx = min(start_idx + dicts_per_process, total_num_list)

    for i in tqdm(range(start_idx, end_idx)):
        wizard_id = total_list[i]['id']
        wizard_q = total_list[i]['conversations'][0]['value']
        wizard_a = total_list[i]['conversations'][1]['value']
        wizard_history = [['Human', wizard_q], ['Assistant', wizard_a]]
            
        url = URL
        data = {
            "history": wizard_history, 
            "temperature": 0.7, 
            "max_new_tokens": 512
        }
        response = requests.post(url, json=data)

        gpt = GPT(user_name='kongchuyi', new_version='0.1.0')
        gpt.post_robot.history_message.append({"role": "system", "content": role_of_chatGPT})
        gpt.post_robot.history_message.append({"role": "user", "content": wizard_q})
        gpt.post_robot.history_message.append({"role": "assistant", "content": wizard_a})

        num_of_rounds = 1
        one_session_dict = {} 
        one_session_list = []
        one_session_list.append({"from": "human", "value": wizard_q})
        one_session_list.append({"from": "gpt", "value": wizard_a})

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
        
        if num_of_rounds == 1:
            logger.info(f'Wizard exceeds, skipped Session ID: {wizard_id}')
            continue

        one_session_dict['session_id'] = wizard_id
        one_session_dict['num_of_rounds'] = num_of_rounds
        one_session_dict['conversations'] = one_session_list

        with open(output_path, 'a', encoding='utf-8') as f:
            json.dump(one_session_dict, f, ensure_ascii=False)
            f.write('\n')
        
if __name__ == '__main__':
    start_time = time.time()

    # For wizardLM, some ids are duplicated
    exist_list = read_json_file(exist_path)
    id_list = []
    for session_dict in exist_list:
        id_list.append(session_dict['session_id'])

    new_list = []
    total_list = read_json_file(wizard_path)
    for session_dict in total_list:
        if session_dict['id'] in id_list:
            continue
        new_list.append(session_dict)
    
    # For some sessions, one round of sessions exceeds the maximal context len
    exceed_list = read_json_file(exceed_path)
    final_list = []
    for session_dict in new_list:
        if session_dict['id'] in exceed_list:
            continue
        final_list.append(session_dict)
    
    logger.info(f'## Retains {len(final_list)} Sessions!!!')
    
    with Pool() as pool:
        pool.starmap(chat, [(i, NUM_OF_PROCESSES, final_list) for i in range(NUM_OF_PROCESSES)])
    
    end_time = time.time()
    print(f'Elapsted {end_time-start_time} seconds.')

