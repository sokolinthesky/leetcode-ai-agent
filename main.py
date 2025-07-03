from enum import Enum, auto
import random
import uuid
from dataclasses import dataclass, fields
from typing import Sequence

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from utils import convert_file_to_object_list, convert_object_list_to_file

load_dotenv()

FILE = "example.md"


class Difficulty(Enum):
    LOW = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    ANY = "Any"


@dataclass
class LeetcodeTask:
    """Leetcode task"""
    id: int
    difficulty: str  # task difficulty
    link: str  # link to leetcode platform
    count: int  # how many times was solved/repeated
    type: str  # type of task, ex: Array, Dynamic programming...
    comment: str  # shot explanation


leetcode_tasks_map = {int(t.id): t for t in convert_file_to_object_list(FILE, LeetcodeTask)}


@tool
def select_random_leetcode_task(difficulty: Difficulty) -> LeetcodeTask:
    """
    Returns random leetcode task

    Args:
        difficulty (Difficulty): required difficulty of the task, use 'Any' if a difficulty does not matter
    """

    filtered_tasks = leetcode_tasks_map if Difficulty.ANY == difficulty else {tid: t for tid, t in
                                                                              leetcode_tasks_map.items() if
                                                                              t.difficulty == difficulty.value}
    key = random.choice(list(filtered_tasks.keys()))
    return filtered_tasks[key]


@tool
def update_leet_code_task(leetcode_task: LeetcodeTask) -> None:
    """
    Updates leetcode task

    Args:
        leetcode_task: (LeetcodeTask) task to update
    """
    print("Updating leetcode task {}".format(leetcode_task))
    leetcode_tasks_map[leetcode_task.id] = leetcode_task
    convert_object_list_to_file(list(leetcode_tasks_map.values()), FILE, [field.name for field in fields(LeetcodeTask)])


tools = [select_random_leetcode_task, update_leet_code_task]


class LLMAgent:
    def __init__(self, model: LanguageModelLike, tools: Sequence[BaseTool]):
        self._model = model
        self._agent = create_react_agent(
            model,
            tools=tools,
            checkpointer=InMemorySaver())
        self._config: RunnableConfig = {
            "configurable": {"thread_id": uuid.uuid4().hex}}  # uniq id for every chat

    def upload_file(self, file):
        file_uploaded_id = self._model.upload_file(file).id_  # type: ignore
        return file_uploaded_id

    def invoke(self, content: str, temperature: float = 0.1) -> str:
        """Sending message to chat"""
        message: dict = {
            "role": "user",
            "content": content
        }
        return self._agent.invoke(
            {
                "messages": [message],
                "temperature": temperature
            },
            config=self._config)["messages"][-1].content


def get_user_prompt() -> str:
    return input("\nYou: ")


def main():
    system_prompt = """
        Your task is to select LeetCode problems.
        
        Default difficulty is 'Any'.
        
        You need to show the "id", "link", "difficulty", and "count".
        Don't print "comment" and "type" of the task until I ask you, just remember them. 
        You should use all task properties if you need this data for the function invocation.
        
        Don't invent any data; strictly use the data returned by function.
        
        Additionally, when I say that I have finished a task, 
        you need to increment the count of that task and update it.
        
        If a task doesn't have "comment," "type," or "difficulty," or if they are empty, 
        you can ask me to fill them in. 

        Now, please give me a random LeetCode task.
    """

    chat = ChatCohere()

    agent = LLMAgent(chat, tools)
    res = agent.invoke(content=system_prompt)

    while (True):
        print(res)
        res = agent.invoke(get_user_prompt())


if __name__ == '__main__':
    main()
