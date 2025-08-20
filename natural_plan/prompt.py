from langchain.prompts import PromptTemplate


FIXED_DATES = """You are an expert in solving complex trip planning problems. Given a specific query, your task is to identify all cities with fixed dates and output the city names along with their corresponding dates.

***** Example *****
[Query]:
You plan to visit 10 European cities for 25 days in total. You only take direct flights to commute between cities. You plan to stay in Berlin for 4 days. You have to attend a workshop in Berlin between day 12 and day 15. You would like to visit Prague for 2 days. You plan to stay in Stuttgart for 5 days. You would like to visit Manchester for 3 days. You want to spend 2 days in Nice. You would like to visit Reykjavik for 2 days. You would like to visit Florence for 3 days. You plan to stay in Vilnius for 5 days. You would like to meet your friends at Vilnius between day 15 and day 19 to tour together. You plan to stay in Oslo for 4 days. You would like to visit Dubrovnik for 4 days. You plan to visit relatives in Dubrovnik between day 1 and day 4.

Here are the cities that have direct flights:
from Reykjavik to Stuttgart, Manchester and Stuttgart, Nice and Berlin, Oslo and Prague, Stuttgart and Berlin, Manchester and Nice, Reykjavik and Oslo, Reykjavik and Prague, Manchester and Prague, Reykjavik and Berlin, Dubrovnik and Manchester, Manchester and Oslo, Manchester and Berlin, Prague and Florence, Berlin and Vilnius, Dubrovnik and Oslo, Nice and Oslo, Berlin and Oslo, Nice and Reykjavik, Vilnius and Oslo.

Find a trip plan of visiting the cities for 25 days by taking direct flights to commute between them.

[Cities with fixed dates]:
Berlin:from day 12 to day 15;Vilnius: from day 15 to day 19;Dubrovnik: from day 1 to day 4
***** Example Ends ****

[Requirements]:
1. You must output all cities with fixed dates
2. Output them in the following strict format: [City Name]: from {{Arrival Date}} to {{Departure Date}}, and separate different cities with a semicolon.

[Query]:
{query}

[Cities with fixed dates]:
"""

EXPANSION = """You excel at solving trip planning problems through multi-step reasoning. Given a specific query and the currently determined planning content, your task is to add a new city and its corresponding dates to the plan.

Requirements:
[1] Expand only once: You are only required to expand one step further based on the current planning content. Do not perform further expansions.
[2] Output format: "[City Name]: from {{Start Date}} to {{End Date}}"
[3] For the [City Name]: 
    [3.1] You can only choose the new city that have DIRECT FLIGHTS to one old city already included in the current planning content, which we refer to the old city as the adjacent city.
    [3.2] Cities cannot be repeated.
    [3.3] Once the adjacent city is fixed, if you are unsure which new city to select, you should list all possible options separated by semicolons. Refer to the example below for guidance.
[4] For the corresponding dates [Day i to Day j]:
    [4.1] The dates for the new city must be consecutive with the dates of the adjacent city.
        [4.1.1] If the new city is before the adjacent city, then the start date of the adjacent city MUST SEARVE AS the end date of the new city.
        [4.1.2] If the new city is after the adjacent city, then the end date of the adjacent city MUST SEARVE AS the start date of the new city.
    [4.2] The date duration of the new city must align with the query's requirements. When calculating the date duration, both the start and end dates should be included. For example, [Day 4 to Day 6] is 3 days.
    [4.3] The selected date duration must be WITHIN the [Available Dates].

***** Example *****
[Query]:
You plan to visit 10 European cities for 25 days in total. You only take direct flights to commute between cities. You plan to stay in Berlin for 4 days. You have to attend a workshop in Berlin between day 12 and day 15. You would like to visit Prague for 2 days. You plan to stay in Stuttgart for 5 days. You would like to visit Manchester for 3 days. You want to spend 2 days in Nice. You would like to visit Reykjavik for 2 days. You would like to visit Florence for 3 days. You plan to stay in Vilnius for 5 days. You would like to meet your friends at Vilnius between day 15 and day 19 to tour together. You plan to stay in Oslo for 4 days. You would like to visit Dubrovnik for 4 days. You plan to visit relatives in Dubrovnik between day 1 and day 4.

Here are the cities that have direct flights:
from Reykjavik to Stuttgart, Manchester and Stuttgart, Nice and Berlin, Oslo and Prague, Stuttgart and Berlin, Manchester and Nice, Reykjavik and Oslo, Reykjavik and Prague, Manchester and Prague, Reykjavik and Berlin, Dubrovnik and Manchester, Manchester and Oslo, Manchester and Berlin, Prague and Florence, Berlin and Vilnius, Dubrovnik and Oslo, Nice and Oslo, Berlin and Oslo, Nice and Reykjavik, Vilnius and Oslo.

Find a trip plan of visiting the cities for 25 days by taking direct flights to commute between them.

[Current Planning]:
Berlin:from day 12 to day 15
Vilnius: from day 15 to day 19
Dubrovnik: from day 1 to day 4

[Available Dates]:
from day 4 to day 12, from day 19 to day 25

[Next City]:
Manchester: from day 4 to day 6; Oslo: from day 4 to day 7
***** Example Ends ****

[Query]:
{query}

[Current Planning]:
{current_chain}

[Available Dates]:
{available_dates}

Now directly output the options without including any additional explanations or comments.
[Next City]:
"""

DECISION =  """You are an expert in solving complex trip planning problems. Given a specific query and several partial plans, your task is to select the top 5 plans that are most likely to succeed in the subsequent steps of reasoning.

Requirements:
1. The date duration of the expanded city must align with the query's requirements. When calculating the date duration, both the start and end dates should be included.
2. No city should be repeated in a plan.
3. Any two cities in a plan that are connected at any time must have a corresponding direct flight.

In addition to the above strict requirements, think through and select the 5 results that you believe are most likely to succeed in the subsequent steps.
Directly output the indices(e.g., 2,3,4,5,6) without including any additional explanations or comments.

Query:
{query}

Candidate Plans:
{candidate_chains}

Output:"""

PLAN_GENERATION = """You are skilled at creating trip plans based on the given information. Given each city and its stay dates, your task is to convert this into the specified planning format.

Double-check that the start and end dates for each city are correct:
1. The date duration of the expanded city must align with the query's requirements. When calculating the date duration, both the start and end dates should be included.
2. The end date of the last city should be the start date of the subsequent city.

***** Example *****
[Query]:
You plan to visit 5 European cities for 23 days in total. You only take direct flights to commute between cities. You plan to stay in Edinburgh for 7 days. You would like to visit Paris for 6 days. You plan to visit relatives in Paris between day 13 and day 18. You would like to visit Riga for 4 days. You would like to meet your friends at Riga between day 1 and day 4 to tour together. You plan to stay in Seville for 4 days. You would like to visit Naples for 6 days.

Here are the cities that have direct flights:
Riga and Paris, Edinburgh and Paris, Edinburgh and Seville, Seville and Paris, Riga and Edinburgh, Paris and Naples.

Find a trip plan of visiting the cities for 23 days by taking direct flights to commute between them.

[Given Information]:
Riga: from day 1 to day 4
Paris: from day 13 to day 18
Edinburgh: from day 4 to day 10
Naples: from day 18 to day 23
Seville: from day 10 to day 13

[Plan]:
**Day 1-4:** Arriving in Riga and visit Riga for 4 days.
**Day 4:** Fly from Riga to Edinburgh.
**Day 4-10:** Visit Edinburgh for 7 days.
**Day 10:** Fly from Edinburgh to Seville.
**Day 10-13:** Visit Seville for 4 days.
**Day 13:** Fly from Seville to Paris.
**Day 13-18:** Visit Paris for 6 days.
**Day 18:** Fly from Paris to Naples.
**Day 18-23:** Visit Naples for 6 days.
***** Example Ends ****

[Query]:
{query}

[Given Information]:
{given_information}

Directly output the plan without including any additional explanations or comments.
[Plan]:
"""

PLAN_DECISION = """You are an expert in solving complex trip planning problems. Given a specific query and several candidate plans, only one solution is correct. Your task is to identify the correct plan.

Query:
{query}

The criterion for determining the correct plan is as follows: Any two cities in the plan that are connected at any point in time must have a direct flight between them.
Candidate Plans:
{candidate_chains}

Directly output the index without including any additional explanations or comments.
Output:
"""

fixed_dates_prompt = PromptTemplate(input_variables=['query'], template=FIXED_DATES)

expansion_prompt = PromptTemplate(input_variables=['query','current_chain','available_dates'], template=EXPANSION)

decision_prompt = PromptTemplate(input_variables=['query','candidate_chains'], template=DECISION)

plan_generation_prompt = PromptTemplate(input_variables=['query', 'given_information'], template=PLAN_GENERATION)

plan_decision_prompt = PromptTemplate(input_variables=['query','candidate_chains'], template=PLAN_DECISION)
