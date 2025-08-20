import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from tqdm import tqdm
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from prompt import fixed_dates_prompt,expansion_prompt,decision_prompt,plan_generation_prompt,plan_decision_prompt
import math
import re

class HyperTree:
    def __init__(self, value):
        self.value = value 
        self.all = []  
        self.branch = None  
        self.children = []  
    def show(self, depth=0):
        result = '<Tab>' * depth +self.value + '\n'
        for child in self.children:
            result += child.show(depth + 1) 
        return result
    def is_leaf(self):
        return len(self.children) == 0

def convert_ai_message_to_dict(ai_message, threshold=0.1):
    logprobs_obj = ai_message.response_metadata['logprobs']
    if type(logprobs_obj) is not dict:
        logprobs_obj = logprobs_obj.to_dict()
    logprobs_content = logprobs_obj['content'][0]['top_logprobs']
    
    token_prob_dict = {}
    for entry in logprobs_content:
        token = entry['token']
        logprob = entry['logprob']
        probability = math.exp(logprob)
        if probability>threshold:
            token_prob_dict[token] = probability
    return token_prob_dict

def convert_ai_message_sequence_to_probility(ai_message):
    sequence_log_probility = 0
    for i in range(ai_message.response_metadata['token_usage'].to_dict()['completion_tokens']):
        sequence_log_probility +=ai_message.response_metadata['logprobs'].to_dict()['content'][0]['logprob']
    sequence_probability = math.exp(sequence_log_probility)
    return sequence_probability

def format_step(step: str) -> str:
    return step.strip('.').strip('\n').strip().replace('\n', '').strip('```')

def dict_to_string(input_dict):
    result = ""
    for key, value in input_dict.items():
        result += f"{key}:{value}\n"
    return result
from vllm import LLM, SamplingParams
class ResponseGenerator:
    def __init__(self, json_file, engine, method, verbose, prompt=fixed_dates_prompt):
        self.engine = engine
        self.verbose = verbose
        self.method = method
        self.fixed_dates_prompt=fixed_dates_prompt
        self.expansion_prompt=expansion_prompt
        self.decision_prompt=decision_prompt
        self.plan_generation_prompt=plan_generation_prompt
        self.plan_decision_prompt=plan_decision_prompt
        self.max_gpt_response_length = 500
        self.data = self.read_json(json_file)
        if self.engine == 'bloom':
            self.model = self.get_bloom()
        else:
            #self.model = LLM(model="/home/shared_data/Qwen2-72B-Instruct",tensor_parallel_size = 8)
            self.model = ChatOpenAI(model_name="/home/shared_data/QwQ-32B-Preview",
                temperature=0,
                max_tokens=4096,
                openai_api_key=os.getenv('OPENAI_API_KEY'), 
                openai_api_base=os.getenv('OPENAI_API_BASE'))
    def find_solution_path(self, node, path=None):
        if path is None:
            path = []
        current_path = path + [node]
        if not node.children:
            return [current_path]
        all_paths = []
        for child in node.children:
            all_paths.extend(self.find_solution_path(child, current_path))
        return all_paths
    def traverse_tree_bfs(self, depth, width):
        for i in range(depth):
            solutions = self.find_solution_path(self.root) 
            if len(solutions)>width:
                self.candidate_chains = {}
                for i,element in enumerate(solutions):
                    tuple_list = []
                    for item in element:
                        value = item.value
                        num_part = int(value.strip().rsplit(' ', 1)[-1])
                        tuple_list.append((num_part, value))
                    tuple_list.sort(key=lambda x: x[0])
                    sorted_value_list = [item[1] for item in tuple_list]
                    candidate_value = (', '.join(value for value in sorted_value_list)).replace('\n',',')
                    if candidate_value not in self.candidate_chains.values():
                        self.candidate_chains[len(self.candidate_chains)] = candidate_value
                indices = format_step(self.model([HumanMessage(content=self._decision_prompt())]).content)
                indices = indices.split(',')
                for i in range(len(indices)):
                    indices[i] = int(indices[i])
                indices = sorted(indices, reverse=True)
                remove_solutions = solutions
                solutions = []
                for index in indices:
                    solutions.append(remove_solutions[index])
                    remove_solutions.pop(index)
                for solution in remove_solutions:
                    for i in range(len(solution) - 1, 0, -1):
                            current_node = solution[i]
                            parent_node = solution[i - 1]
                            if current_node in parent_node.children:
                                parent_node.children.remove(current_node)
                            if len(parent_node.children) == 0:
                                continue  
                            else:
                                break 
            for solution in solutions:
                leaf = solution[-1]
                number = []
                for node in solution:
                    words = node.value.strip().split(' ')
                    start = int(words[-4])
                    end = int(words[-1])
                    if start in number:
                        number.remove(start)
                    else:
                        number.append(start)
                    if end in number:
                        number.remove(end)
                    else:
                        number.append(end)
                if 1 in number:
                    number.remove(1)
                else:
                    number.append(1)
                if self.total_dates in number:
                    number.remove(self.total_dates)
                else:
                    number.append(self.total_dates)
                number.sort()
                self.available_dates = ""
                for i in range(0, len(number), 2):
                    if i + 1 < len(number):
                        s = f'from day {number[i]} to day {number[i + 1]}; '
                        self.available_dates+=s
                self.current_chain = ('\n'.join(node.value for node in solution)).replace('\\n','\n')
                request = format_step(self.model([HumanMessage(content=self._expansion_prompt())]).content)
                try:
                    if ';' in request:
                        parts = request.split(';')
                        for part in parts:
                            city, days = part.split(':')
                            city = city.strip()
                            days = days.strip()
                            leaf.children.append(HyperTree(f"{city}: {days}"))
                    else:
                        city, days = request.split(':')
                        city = city.strip()
                        days = days.strip()
                        leaf.children.append(HyperTree(f"{city}: {days}"))
                except:
                    pass



    def read_json(self, json_file):
        with open(json_file, 'r') as file:
            return json.load(file)
        
    def get_bloom(self):
        max_memory_mapping = {0: "0GB", 1: "43GB", 2: "43GB", 3: "43GB", 4: "43GB", 5: "43GB"}
        cache_dir = os.getenv('BLOOM_CACHE_DIR', '/data/karthik/LLM_models/bloom/')
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", cache_dir=cache_dir,
                                                     local_files_only=False, load_in_8bit=True, device_map='auto',
                                                     max_memory=max_memory_mapping)
        return {'model': model, 'tokenizer': tokenizer}

    def get_responses(self):
        i = 0
        for instance in tqdm(list(self.data.values())[:]):
            if not ((0 <= i < 20) or (200 <= i < 220) or (400 <= i < 420) or (600 <= i < 620) or (800 <= i < 820) or (1000 <= i < 1020) or (1200 <= i < 1220) or (1400 <= i < 1420)):
                i += 1
                continue
            if self.verbose:
                print(f"Sending query to LLM: Instance trip_planning_example_{i}")
            #query = instance["prompt_0shot"]
            ours = False
            if ours:
                self.query = query
                days = instance['durations'].split('**')
                self.total_dates = 1-len(days)
                for day in days:
                    self.total_dates += int(day)
                error_count = 0
                while error_count < 3:
                    try:
                        request = self.model([HumanMessage(content=self._fixed_dates_prompt())]).content
                        parts = request.split(';')
                        city, days = parts[0].split(':')
                        city = city.strip()
                        days = days.strip()
                        self.root = HyperTree(f"{city}: {days}")
                        self.leaf = self.root
                        for part in parts[1:]:
                            city, days = part.split(':')
                            city = city.strip()
                            days = days.strip()
                            self.leaf.children.append(HyperTree(f"{city}: {days}"))
                            self.leaf = self.leaf.children[0]
                        depth = int(instance['num_cities'])-len(parts)
                        break
                    except Exception as e:
                        error_count += 1
                        print(f"An error occurred: {e}. Retrying... ({error_count}/3)")

                self.traverse_tree_bfs(depth = depth, width = 10)
                solutions = self.find_solution_path(self.root)
                self.candidate_chains = {}
                for i,element in enumerate(solutions):
                    tuple_list = []
                    for item in element:
                        value = item.value
                        num_part = int(value.strip().rsplit(' ', 1)[-1])
                        tuple_list.append((num_part, value))
                    tuple_list.sort(key=lambda x: x[0])
                    sorted_value_list = [item[1] for item in tuple_list]
                    candidate_value = (', '.join(value for value in sorted_value_list)).replace('\n',',')
                    if candidate_value not in self.candidate_chains.values():
                        self.candidate_chains[len(self.candidate_chains)] = candidate_value
                
                indice = format_step(self.model([HumanMessage(content=self._plan_decision_prompt())]).content)
                solution = solutions[int(indice)]
                self.solution = ('\n'.join(node.value for node in solution)).replace('\\n','\n')
                llm_response = self.model([HumanMessage(content=self._plan_generation_prompt())]).content
                if not llm_response:
                    print(f"Failed instance: trip_planning_example_{i}")
                    continue
                if self.verbose:
                    print(f"LLM response: {llm_response}")
                i += 1
            else:
                llm_response = self.model([HumanMessage(content=instance["prompt_5shot"])]).content
            instance["pred_5shot_pro"] = llm_response
            i += 1
            with open('./trip_qwen.json', 'w') as file:
                json.dump(self.data, file, indent=4)

    def _fixed_dates_prompt(self) -> str:
        return self.fixed_dates_prompt.format(query = self.query) 
    def _expansion_prompt(self) -> str:
        return self.expansion_prompt.format(query = self.query,current_chain = self.current_chain,available_dates = self.available_dates) 
    def _decision_prompt(self) -> str:
        return self.decision_prompt.format(query = self.query,candidate_chains = dict_to_string(self.candidate_chains)) 
    def _plan_generation_prompt(self) -> str:
        return self.plan_generation_prompt.format(query = self.query, given_information = self.solution) 
    def _plan_decision_prompt(self) -> str:
        return self.plan_decision_prompt.format(query = self.query,candidate_chains = dict_to_string(self.candidate_chains)) 


if __name__=="__main__":
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, help='Task to run \
    \n tor_one_shot = One shot Tree of Reasoning Plan Generation\
    \n baseline_one_shot = One Shot Plan Generation\
    ')
    parser.add_argument('--engine', type=str, required=True, help='Engine to use \
                        \n gpt-4o = GPT-4o \
                        \n gpt-4-turbo = GPT-4-turbo \
                        \n bloom = Bloom \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        \n davinci = GPT-3 Davinci \
                        \n curie = GPT-3 Curie \
                        \n babbage = GPT-3 Babbage \
                        \n ada = GPT-3 Ada \
                        ')
                        
    parser.add_argument('--verbose', type=str, default="True", help='Verbose')
    #config
    parser.add_argument('--task', type=str, required=True, help='trip_planning \n meeting_planning \n calendar_scheduling')
    parser.add_argument('--run_till_completion', type=str, default="False", help='Run till completion')
    args = parser.parse_args()
    task = args.task
    engine = args.engine
    method = args.method
    verbose = eval(args.verbose)

    print(f"Task: {task}, Engine: {engine}, Method: {method}, Verbose: {verbose}")
    json_file = f'./data/{task}.json'
    response_generator = ResponseGenerator(json_file, engine, method, verbose)
    response_generator.get_responses()