import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt,slow_thinking_prompt, react_planner_agent_prompt,reflect_prompt,react_reflect_planner_agent_prompt,extract_value_prompt,\
      REFLECTION_HEADER, PLAN_CONVERT, THOUGHT_CONVERT
from agents.hypertree_prompts import select_prompt, expand_prompt,decide_prompt,execute_prompt,convert_prompt,tree_convert_prompt,plan_generation_prompt
from evaluation.eval import reflect_plan
from langchain_community.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json
from env import ReactEnv,ReactReflectEnv
import tiktoken
import re
import openai
import time
from enum import Enum
from typing import List, Union, Literal,Optional
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from postprocess.openai_request import build_plan_format_conversion_prompt,prompt_chatgpt
from func import remove_lines
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
MAX_ITERATIONS = 30

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


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 # args,
                 agent_prompt: PromptTemplate = planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        max_token_length=30000

        if model_name in  ['mistral-7B-32K']:
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8301/v1", 
                     model_name="gpt-3.5-turbo")
        
        elif model_name in  ['ChatGLM3-6B-32K']:
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="gpt-3.5-turbo")
            
        elif model_name in ['mixtral']:
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="YOUR/MODEL/PATH")
            
        elif model_name in ['gemini']:
            self.llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        elif model_name in ['gpt-4o']:
            self.max_token_length = max_token_length
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=4096,
                openai_api_key="sk-K02g41hRu880wfs7aWlFAcpEY10futI78FT7Cwp6XUBwNArQ", 
                openai_api_base="https://api2.aigcbest.top/v1", 
                model_name="gpt-4o"
            )
        elif model_name == 'gpt-3.5-turbo':
            self.max_token_length = max_token_length
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8, max_tokens=4096, openai_api_key=OPENAI_API_KEY)


        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, number, log_file=None) -> str:
        if log_file:
            log_file.write('\n---------------Planner\n'+self._build_agent_prompt(text, query))
        #print(self._build_agent_prompt(text, query))
        if 'gpt' in self.model_name:
            import ipdb;ipdb.set_trace()
            request = self.llm([HumanMessage(content=self._build_agent_prompt(text, query))])
            return request.content
        elif self.model_name in ['gemini']:
            return str(self.llm.invoke(self._build_agent_prompt(text, query)).content)
        else:
            if len(self.enc.encode(self._build_agent_prompt(text, query))) > 12000:
                return 'Max Token Length Exceeded.'
            else:
                request = self.llm([HumanMessage(content=self._build_agent_prompt(text, query))]).content
                return request

    def _build_agent_prompt(self, text, query) -> str:
        return self.agent_prompt.format(
            given_information=text,
            query=query)


class ReactPlanner:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        self.agent_prompt = agent_prompt
        self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation"]})
        self.env = ReactEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False


class ReactReflectPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
                reflect_prompt: PromptTemplate = reflect_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        if model_name in ['gemini']:
            self.react_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
            self.reflect_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        else:
            self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
            self.reflect_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
        self.model_name = model_name
        self.env = ReactReflectEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
            if self.env.is_terminated and not self.finished:
                self.reflect(ReflexionStrategy.REFLEXION)

        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_agent(self) -> str:
        while True:
            try:
                if self.model_name in ['gemini']:
                    return format_step(self.react_llm.invoke(self._build_agent_prompt()).content)
                else:
                    return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def prompt_reflection(self) -> str:
        while True:
            try:
                if self.model_name in ['gemini']:
                    return format_step(self.reflect_llm.invoke(self._build_reflection_prompt()).content)
                else:
                    return format_step(self.reflect_llm([HumanMessage(content=self._build_reflection_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_reflection_prompt())
                print(len(self.enc.encode(self._build_reflection_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad,
                            reflections = self.reflections_str)
    
    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False
        self.reflections = []
        self.reflections_str = ''
        self.env.reset()



class HyperTree:
    def __init__(self, value):
        self.value = value
        self.all = []
        self.branch = None
        self.children = []  
    def show(self, depth=0):
        result = '<Tab>' * depth + self.value + '\n'
        for child in self.children:
            result += child.show(depth + 1) 
        return result
    def is_terminal(self):
        self.non_terminals = ["[Plan]", "[Transportation]", "[Taxi]", "[Self-driving]", "[Flight]",
            "[Accommodation]", "[Attraction]", "[Dining]"]
        self.terminals = ["[environment]", "[preference]", "[cost]", "[consistency]", 
            "[house rule]", "[room type]", "[minimum stay]", "[cuisine]"]
        node = self.value.lower()
        if any(nt.lower() == node for nt in self.non_terminals):
            return False
        if any(t.lower() == node for t in self.terminals):
            return True
        wrong_patter = re.compile(r'\[(.*?)\]\s*(.*?)\s*\[(.*?)\]')
        self.transportation_pattern = re.compile(r'\[transportation from [\w\s.]+ to [\w\s.]+\]', re.IGNORECASE)
        self.accommodation_pattern = re.compile(r'\[accommodation for [\w\s.]+\]', re.IGNORECASE)
        self.dining_pattern = re.compile(r'\[dining for [\w\s.]+\]', re.IGNORECASE)
        self.attraction_pattern = re.compile(r'\[attraction for [\w\s.]+\]', re.IGNORECASE)
        if wrong_patter.match(node):
            return True
        if self.transportation_pattern.match(node):
            return False
        elif self.accommodation_pattern.match(node):
            return False
        elif self.dining_pattern.match(node):
            return False
        elif self.attraction_pattern.match(node):
            return True
        return True
    def is_leaf(self):
        return len(self.children) == 0
    def get_leaves(self):
        leaves = []
        if self.is_leaf():
            leaves.append(self)
        else:
            for child in self.children:
                leaves.extend(child.get_leaves())
        return leaves
    def postorder_traversal(self):
        for child in self.children:
            is_found = child.postorder_traversal()
            if is_found:
                return True
        if self.branch == None:
            return False
        if self.branch+1 < len(self.all):
            self.branch = self.branch+1
            children_value_list=self.all[self.branch]
            self.children = [HyperTree(value) for value in children_value_list]
            return True
        return False
    


class HTPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """
    def __init__(self,
                select_prompt: PromptTemplate = select_prompt,
                expand_prompt: PromptTemplate = expand_prompt,
                decide_prompt: PromptTemplate = decide_prompt,
                execute_prompt: PromptTemplate = execute_prompt,
                model_name: str = 'gpt-4o',
                ) -> None:
        self.select_prompt = select_prompt
        self.expand_prompt = expand_prompt
        self.convert_prompt = convert_prompt
        self.decide_prompt = decide_prompt
        self.execute_prompt = execute_prompt
        self.tree_convert_prompt = tree_convert_prompt
        self.transportation_pattern = re.compile(r'\[transportation from [\w\s.]+ to [\w\s.]+\]', re.IGNORECASE)
        self.accommodation_pattern = re.compile(r'\[accommodation for [\w\s.]+\]', re.IGNORECASE)
        self.dining_pattern = re.compile(r'\[dining for [\w\s.]+\]', re.IGNORECASE)
        self.attraction_pattern = re.compile(r'\[attraction for [\w\s.]+\]', re.IGNORECASE)
        if model_name in ['gemini']:
            self.plan_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
            self.execute_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        else:
            self.plan_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=4096, openai_api_key=OPENAI_API_KEY)
            self.execute_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=4096, openai_api_key=OPENAI_API_KEY)
            self.convert_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=4096, openai_api_key=OPENAI_API_KEY)
        
        self.model_name = model_name
        self.query = None
        self.finished = False
        self.enc = tiktoken.encoding_for_model("gpt-4o")
    def select(self,node):
        leaves = node.get_leaves()
        self.leaves = []
        for leaf in leaves:
            if not leaf.is_terminal():
                self.leaves.append(leaf)
        self.leaves_dict = {i: leaf.value for i, leaf in enumerate(self.leaves)}
        if self.leaves == []:
            return False
        #request = format_step(self.plan_llm([HumanMessage(content=self._build_select_prompt())]).content)
        #print(request)
        return '0'
    def generate_responses(self, category):
        if self.visiting_city_number == 1:
            responses = {
                "[Transportation]": f"[Transportation from {self.org} to {self.dest}][Transportation from {self.dest} to {self.org}]",
                "[Accommodation]": f"[Accommodation for {self.dest}]",
                "[Attraction]": f"[Attraction for {self.dest}]",
                "[Dining]": f"[Dining for {self.dest}]"
            }
        elif self.visiting_city_number == 2:
            responses = {
                "[Transportation]": f"[Transportation from {self.org} to City 1 in {self.dest}][Transportation from City 1 in {self.dest} to City 2 in {self.dest}][Transportation from City 2 in {self.dest} to {self.org}]",
                "[Accommodation]": f"[Accommodation for City 1 in {self.dest}][Accommodation for City 2 in {self.dest}]",
                "[Attraction]": f"[Attraction for City 1 in {self.dest}][Attraction for City 2 in {self.dest}]",
                "[Dining]": f"[Dining for City 1 in {self.dest}][Dining for City 2 in {self.dest}]"
            }
        elif self.visiting_city_number == 3:
            responses = {
                "[Transportation]": f"[Transportation from {self.org} to City 1 in {self.dest}][Transportation from City 1 in {self.dest} to City 2 in {self.dest}][Transportation from City 2 in {self.dest} to City 3 in {self.dest}][Transportation from City 3 in {self.dest} to {self.org}]",
                "[Accommodation]": f"[Accommodation for City 1 in {self.dest}][Accommodation for City 2 in {self.dest}][Accommodation for City 3 in {self.dest}]",
                "[Attraction]": f"[Attraction for City 1 in {self.dest}][Attraction for City 2 in {self.dest}][Attraction for City 3 in {self.dest}]",
                "[Dining]": f"[Dining for City 1 in {self.dest}][Dining for City 2 in {self.dest}][Dining for City 3 in {self.dest}]"
            }
        return responses[category]

    def expand(self, node):
        if node.value in ["[Transportation]", "[Accommodation]", "[Attraction]", "[Dining]"]:
            request =  self.generate_responses(node.value)
        elif self.transportation_pattern.fullmatch(node.value):
            request =  self.rules['[Transportation from A to B]']
        elif self.accommodation_pattern.fullmatch(node.value):
            request =  self.rules['[Accommodation for A]']
        elif self.dining_pattern.fullmatch(node.value):
            request =  self.rules['[Dining for A]']
        elif node.value in self.rules:
            request =  self.rules[node.value]
        try: 
            request = ['[' + item + ']' for item in request.strip('[]').split('][')]
        except:
            self.str_to_convert = request
            request = format_step(self.convert_llm([HumanMessage(content=self._build_convert_prompt())]).content)
            request = ['[' + item + ']' for item in request.strip('[]').split('][')]

        node.all = [request]
        node.branch = 0
        children_value_list=node.all[node.branch]
        for value in children_value_list:
            if value!= node.value:
                node.children.append(HyperTree(value))
    
    def run(self, given_information, query, number) -> None:
        self.given_information = given_information
        self.current_tree = HyperTree('[Plan]')
        self.selected_node = self.current_tree
        self.number = number
        self.query = query['query']
        self.dest = query['dest']
        self.visiting_city_number = query['visiting_city_number']
        self.org = query['org']
        self.rules = {'[Plan]':'[Transportation][Accommodation][Attraction][Dining]','[Transportation from A to B]':'[Self-driving][Taxi][Flight]',\
                      '[Self-driving]':'[transportation availability][transportation preference][transportation cost]',\
                        '[Taxi]':'[transportation availability][transportation preference][transportation cost]',\
                            '[Flight]':'[transportation availability][transportation preference][transportation cost]',\
                                '[Accommodation for A]':'[minimum stay][house rule][room type][accommodation cost]',\
                                    '[Dining for A]':'[cuisine][dining cost]'}
        iteration_count = 0
        while True:
            if iteration_count >= MAX_ITERATIONS:
                break
            if not self.selected_node:
                branch = self.current_tree.postorder_traversal()
                if not branch:
                    break
            else:
                self.expand(self.selected_node)
            
            selected_index = self.select(self.current_tree)
            if self.leaves==[]:
                break
            self.selected_node = self.leaves[int(selected_index)]
            iteration_count += 1
        #self.final_tree = self.convert_llm([HumanMessage(content=self._build_tree_convert_prompt())]).content
        self.final_tree = self.current_tree.show().rstrip('\n')
        print("超树结构：", self.final_tree)
        self.thinking_process = self.plan_llm([HumanMessage(content=self._build_execute_prompt())]).content
        print("思考过程：",self.thinking_process)
        travel_plan = self.plan_generate()
        return travel_plan

    def plan_generate(self):
        self.plan_generation_prompt = plan_generation_prompt
        plan = self.plan_llm([HumanMessage(content=self._build_plan_generation_prompt())]).content
        print("规划结果：",plan)
        plan_convert_prompt = PLAN_CONVERT+"\nTEXT:\n"+plan+"\nJSON:\n"
        plan = self.convert_llm([HumanMessage(content=plan_convert_prompt)]).content
        plan = plan.lstrip("```json").rstrip('```')
        plan = plan.replace('From','from').replace('back to', 'to')
        plan = "".join(x for x in plan.split("\n"))
        plan = json.loads(plan)
        plan = {"idx": self.number, "query": self.query, "plan": plan}
        return plan
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished
    def _build_select_prompt(self) -> str:
        return self.select_prompt.format(
            query = self.query,
            current_tree = self.current_tree.show().rstrip('\n'),
            leaves = self.leaves_dict)
    
    def _build_expand_prompt(self) -> str:
        return self.expand_prompt.format(
            query = self.query,
            current_tree = self.current_tree.show().rstrip('\n'),
            selected_node = self.selected_node.value)
    
    def _build_convert_prompt(self) -> str:
        return self.convert_prompt.format(
            str_to_convert = self.str_to_convert)
    def _build_tree_convert_prompt(self) -> str:
        return self.tree_convert_prompt.format(
            tree_to_convert = self.current_tree.show().rstrip('\n'))
    def _build_execute_prompt(self) -> str:
        return self.execute_prompt.format(
            given_information = self.given_information,
            query = self.query,
            solution_strategy = self.final_tree)
    def _build_plan_generation_prompt(self) -> str:
        return self.plan_generation_prompt.format(
            query = self.query,
            thinking_process = self.thinking_process)


def format_step(step: str) -> str:
    return step.strip('.').strip('\n').strip().replace('\n', '').strip('```')

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
