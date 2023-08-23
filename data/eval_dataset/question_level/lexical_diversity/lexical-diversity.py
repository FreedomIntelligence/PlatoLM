'''
pip install lexicalrichness
python -m lexical-diversity   
'''
import json

from tqdm import tqdm
from lexicalrichness import LexicalRichness


def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data

def merge_all_utter(session_list: list):
    all_q = []
    for session_dict in session_list:
        conv_list = session_dict['conversations']
        for utter_dict in conv_list:
            if utter_dict['from'] == 'human':
                all_q.append(utter_dict['value'])
    return all_q

def compute_avg_mtld(path: str):
    session_list = read_json_file(path)
    all_q = merge_all_utter(session_list)
    all_q_ld = []
    cnt_zero = 0
    for q in tqdm(all_q):
        lex = LexicalRichness(q)
        try:
            MTLD = lex.mtld(threshold=0.72)
        except ZeroDivisionError as _:
            print(q)
            cnt_zero += 1
            continue
        all_q_ld.append(MTLD)
    avg_mtld = sum(all_q_ld) / len(all_q_ld)
    print(f'Total Question Counts: {len(all_q)}')
    print(f'Skipped {cnt_zero} Questions with Zero Length.')
    print(f'Average MTLD: {avg_mtld}')

if __name__ == "__main__":
    compute_avg_mtld('./ww_10k.json')





