import json
import requests
from tqdm import tqdm

# This is our lab's own api.
class Detector_api():
    def __init__(self, endpoint="predict_all", file_path = None):
        self.API_URL = "http://10.26.1.137:8000"
        self.endpoint = endpoint
        self.file_path = file_path
        
    def batch_predict(self, texts):
        payload = {"texts": texts}
        headers = {"Content-Type": "application/json"}

        response = requests.post(f"{self.API_URL}/{self.endpoint}", data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            results = response.json()
            return results
        else:
            raise Exception(f"Request failed with status code {response.status_code}")
        
def read_data(file_path: str):
    all_of_questions = []
    with open(file_path, 'r', encoding="utf-8") as file:
        session_list = json.load(file)
    for i in tqdm(range(len(session_list)), desc='Preprocessing data..'):
        conv = session_list[i]['conversations']
        for idx, utter_dict in enumerate(conv):
            if idx % 2 == 0:
                all_of_questions.append(utter_dict['value'])
    print(f'Within the {len(session_list)} sessions, {len(all_of_questions)} questions are contained in total.')
    avg_rounds = round(len(all_of_questions)/len(session_list), 2)
    print(f'The average rounds is {avg_rounds}.')
    return all_of_questions

def classification(output_list: list) -> str:
    max_scores = -100
    for label_dict in output_list:
        if label_dict['score'] > max_scores:
            max_scores = label_dict['score']
            max_scores_label = label_dict['label']
    return max_scores_label

def compute_humanlike_ratio(path: str):
    detector = Detector_api(endpoint='predict')

    label_counter = dict()
    all_questions = read_data(path)          

    all_response = []
    for batch in tqdm(range(0, len(all_questions), 1000), desc='Deriving response..'):
        batch_of_questions = all_questions[batch: batch+1000]
        response = detector.batch_predict(batch_of_questions) 
        all_response.extend(response) 
    assert len(all_response) == len(all_questions)

    for q_res_list in tqdm(all_response, desc = 'Calculating human-like ratio..'):
        label = classification(q_res_list)
        if label == 'Human':
            label_counter['Human'] = label_counter.get('Human', 0) + 1
        else:
            label_counter['ChatGPT'] = label_counter.get('ChatGPT', 0) + 1
    print(label_counter)
    
    if 'Human' in label_counter:
        human_likeness_ratio = round(label_counter['Human']/len(all_questions), 4)
    else:
        human_likeness_ratio = 0
    return human_likeness_ratio
    
if __name__ == "__main__":
    hlr = compute_humanlike_ratio('./ww_10k.json')
    print(f'The human-likeness ratio is {hlr}.')
