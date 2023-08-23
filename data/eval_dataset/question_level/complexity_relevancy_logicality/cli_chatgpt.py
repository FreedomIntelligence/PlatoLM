import time
import json
import logging

from tqdm import tqdm
from multiprocessing import Pool, Lock
# this package is only valid in our lab's private internet 
from gpt import GPT


logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_file = './check.log'
file_handler = logging.FileHandler(log_file, mode='a')  
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# set your own pace and name of files
BATCH_SIZE = 100
write_into_file = '.json'
read_from_file = '.json'

unique_ids = set()
unique_lock = Lock()

def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data

def read_txt_file(file_path='./prompt_template.txt'):
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read()
    return content

def generate_eval_list(data_path):
    to_eval_list = [] 
    session_id = 0
    data_list = read_json_file(data_path)

    for session_dict in data_list:
        to_eval_dict = dict()
        session_str = ''

        session_str += '#[Start of multi-round conversation]#\n\n'
        for utter_dict in session_dict['conversations']:
            if utter_dict['from'] == 'human':
                session_str += '#[Human\'s question]#\n'
                session_str += utter_dict['value'] + '\n'
            else:
                session_str += '#[Assistant\'s answer]#\n'
                session_str += utter_dict['value'] + '\n\n'
        session_str += '#[End of multi-round conversation]#\n\n'

        to_eval_dict['id'] = session_id 
        to_eval_dict['conversations'] = session_str
        to_eval_list.append(to_eval_dict)

        session_id += 1
    
    return to_eval_list

def send_request(session_id:int, req:str):  
    gpt = GPT(user_name="kongchuyi", model_name="gpt-3.5-turbo-16k", new_version='0.1.0')

    rep = None
    err = 1
    while err or (not rep) or (rep == 'APIKey Error') or ('invalid_api_key' in rep) or ('account_deactivated' in rep) or ('rate_limit_exceeded' in rep) or ('insufficient_quota' in rep) or ('获取请求参数user_name和parameters失败' in rep) or ('请求官方API失败。Error code:200') in rep:
        try:
            _, rep = gpt.call(req)
            if rep:
                if 'context_length_exceeded' in rep: 
                    logger.info(f'Skipped {session_id} since exceeding the context length.')
                    return None
            err = 0
        except BaseException as e:  
            print(f'\nThe {session_id}-session: \n {type(e)} \n {e}')
    return rep
    
def write_to_file(results):
    with open(write_into_file, 'a', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
        f.write('\n')

def process_batch(batch:list):
    prompt = read_txt_file()

    for session_dict in tqdm(batch):
        if session_dict['id'] not in unique_ids:
            with unique_lock:
                unique_ids.add(session_dict['id'])

            result = send_request(session_dict['id'], prompt+session_dict['conversations']) 
            
            if result:
                result_dict = {
                    'id': session_dict['id'],
                    'scores': result
                }
                write_to_file(result_dict)

if __name__ == '__main__':
    start_time = time.time()

    to_eval_list = generate_eval_list(read_from_file)

    with Pool(processes=12) as pool:
        for i in range(0, len(to_eval_list), BATCH_SIZE):
            batch_list = to_eval_list[i:i+BATCH_SIZE]
            pool.apply_async(process_batch, (batch_list,))
        pool.close()
        pool.join()

    end_time = time.time()
    print(f'Elapsted {end_time-start_time} seconds.')
