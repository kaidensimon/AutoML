import functools
from langchain.agents import AgentExecutor, create_openai_tools_agent
import operator
OUTPUT_Directory = r'D:\fastaimodelmaker\output'
from tools.inspect_dir import inspect_data_dir
from tools.web import research
from tools.image_inspector import inspect_image_properties
from tools.run import run_model
from main import set_environment_variables
from langchain_openai import ChatOpenAI
from CODER_AGENT_PROMPTS import(
AGENT_0_SYSTEM_PROMPT,
AGENT_1_SYSTEM_PROMPT,
AGENT_2_SYSTEM_PROMPT,
AGENT_3_SYSTEM_PROMPT,
AGENT_4_SYSTEM_PROMPT
)
SAVE_PYTHON_FILE_NODE_NAME = "save_file"

from typing import Annotated, Sequence, TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import functools
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
import asyncio
import operator
import uuid

set_environment_variables("FastAI_Generator")

LLM = ChatOpenAI(model="gpt-3.5-turbo-0125")

TAVILY_TOOL = TavilySearchResults(max_results=1)
AGENT_0_NAME = "agent_0"
AGENT_1_NAME = "agent_1"
AGENT_2_NAME = "agent_2"
AGENT_3_NAME = "agent_3"
AGENT_4_NAME = "agent_4"

def create_agent(llm: BaseChatModel, tools: list, system_prompt:str):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def agent_node(state: AgentState, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

async def async_agent_node(state: AgentState, agent, name):
    result = await agent.ainvoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def python_file_node(state: AgentState):
    python_script_content = str(state["messages"][-1].content)
    ja = python_script_content.replace("```", " ")
    ha = ja.replace("python", " ")
    filename = f"{OUTPUT_Directory}/{uuid.uuid4()}.py"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(ha)
    return{
        "messages": [
            HumanMessage(
                content = f"Output written successfully to {filename}",
                name=SAVE_PYTHON_FILE_NODE_NAME
            )
        ]
    }

agent_0  = create_agent(llm=LLM, tools=[TAVILY_TOOL], system_prompt=AGENT_0_SYSTEM_PROMPT)

agent_0_node = functools.partial(
    agent_node, agent=agent_0, name=AGENT_0_NAME
)

agent_1 = create_agent(llm=LLM, tools=[research, inspect_data_dir], system_prompt=AGENT_1_SYSTEM_PROMPT)

agent_1_node = functools.partial(
    async_agent_node, agent=agent_1, name=AGENT_1_NAME
)

agent_2  = create_agent(llm=LLM, tools=[inspect_image_properties], system_prompt=AGENT_2_SYSTEM_PROMPT)

agent_2_node = functools.partial(
    agent_node, agent=agent_2, name=AGENT_2_NAME
)

agent_3  = create_agent(llm=LLM, tools=[inspect_data_dir], system_prompt=AGENT_3_SYSTEM_PROMPT)

agent_3_node = functools.partial(
    agent_node, agent=agent_3, name=AGENT_3_NAME
)

agent_4  = create_agent(llm=LLM, tools=[run_model], system_prompt=AGENT_4_SYSTEM_PROMPT)

agent_4_node = functools.partial(
    agent_node, agent=agent_4, name=AGENT_4_NAME
)


workflow = StateGraph(AgentState)

workflow.add_node(AGENT_0_NAME, agent_0_node)
workflow.add_node(AGENT_1_NAME, agent_1_node)
workflow.add_node(AGENT_2_NAME, agent_2_node)
workflow.add_node(AGENT_3_NAME, agent_3_node)
workflow.add_node(AGENT_4_NAME,agent_4_node)
workflow.add_node(SAVE_PYTHON_FILE_NODE_NAME, python_file_node)

workflow.add_edge(AGENT_0_NAME, AGENT_1_NAME)
workflow.add_edge(AGENT_1_NAME, AGENT_2_NAME)
workflow.add_edge(AGENT_2_NAME, AGENT_3_NAME)
workflow.add_edge(AGENT_3_NAME, SAVE_PYTHON_FILE_NODE_NAME)
workflow.add_edge(SAVE_PYTHON_FILE_NODE_NAME, AGENT_4_NAME)
workflow.add_edge(AGENT_4_NAME, END)
workflow.set_entry_point(AGENT_0_NAME)

fastai_graph = workflow.compile()

async def run_research_graph(input):
    async for output in fastai_graph.astream(input):
        for node_name, output_value in output.items():
            print("_____")
            print(f"Output from node '{node_name}':")
            print(output_value)
        print("\n---\n")


test_input = {"messages": [HumanMessage(content="https://www.tensorflow.org/tutorials/images/classification")]}

asyncio.run(run_research_graph(test_input))