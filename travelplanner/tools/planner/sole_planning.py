import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt,react_reflect_planner_agent_prompt,reflect_prompt, slow_thinking_prompt, slow_reflect_prompt,extract_value_prompt
# from utils.func import get_valid_name_city,extract_before_parenthesis, extract_numbers_from_filenames
import json
import time
from langchain_community.callbacks import get_openai_callback
import pickle
from tqdm import tqdm
from tools.planner.apis import Planner, ReactPlanner, ReactReflectPlanner, HTPlanner
import openai
import argparse
from datasets import load_dataset

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--reflection", type=bool, default=False)
    #parser.add_argument("--use_reflect", type=bool, default=False)
    #parser.add_argument("--use_extract", type=str, default=True)
    args = parser.parse_args()
    directory = f'{args.output_dir}/{args.set_type}'
    if args.set_type == 'train':
        #query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
        data_file_path = './train_data.pkl'
        with open(data_file_path, 'rb') as file:
            query_data_list = pickle.load(file) 
    elif args.set_type == 'validation':
        data_file_path = './validation_data.pkl'
        with open(data_file_path, 'rb') as file:
            query_data_list = pickle.load(file) 
    elif args.set_type == 'test':
        #query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
        data_file_path = './test_data.pkl'
        with open(data_file_path, 'rb') as file:
            query_data_list = pickle.load(file) 
    numbers = [i for i in range(1,len(query_data_list)+1)]
    
    if args.strategy == 'direct':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)
    elif args.strategy == 'cot':
        planner = Planner(model_name=args.model_name, agent_prompt=cot_planner_agent_prompt)
    elif args.strategy == 'react':
        planner = ReactPlanner(model_name=args.model_name, agent_prompt=react_planner_agent_prompt)
    elif args.strategy == 'reflexion':
        planner = ReactReflectPlanner(model_name=args.model_name, agent_prompt=react_reflect_planner_agent_prompt,reflect_prompt=reflect_prompt)
    elif args.strategy == 'hypertree':
        planner = HTPlanner(model_name=args.model_name)

    with get_openai_callback() as cb:
        for number in tqdm(numbers[:]):
            query_data = query_data_list[number-1]
            reference_information = query_data['reference_information']
            result_list = []
            if args.strategy in ['slow']:
                travel_plan  = planner.run(given_information = reference_information,query_data = query_data, number = number)   
            elif args.strategy in ['direct']:
                travel_plan  = planner.run(text = reference_information,query = query_data['query'], number = number)
            elif args.strategy in ['react','reflexion']:
                planner_results, scratchpad  = planner.run(query_data['query'],given_information = reference_information)
            elif args.strategy in ['hypertree']:
                planner.run(reference_information,query_data,number)
            else:
                planner_results  = planner.run(query_data['query'],given_information = reference_information)
        # check if the directory exists
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
            os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
            result =  [{}]
        else:
            result = json.load(open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')))
        if args.strategy in ['react','reflexion']:
            result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results_logs'] = scratchpad 
        result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results'] = planner_results
        # write to json file
        with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
            json.dump(result, f, indent=4)
