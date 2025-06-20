import random
import uuid
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

FILE = "example.md"


@dataclass
class LeetcodeTask:
    """Leetcode task"""
    id: int
    link: str  # link to leetcode platform
    difficulty: str  # task difficulty
    count: str  # how many times was solved/repeated
    type: str  # type of task, ex: Array, Dynamic programming...
    comment: str  # shot explanation


@tool
def select_random_leetcode_task() -> LeetcodeTask:
    """Returns random leetcode task"""
    leetcode_tasks = convert_file_to_object_list()
    random_task = random.choice(leetcode_tasks)
    return random_task


def update_leet_code_task(task_id: int, difficulty: str, count: int, type: str, comment: str) -> None:
    """Updates leetcode task"""
    None


def convert_file_to_object_list() -> list[LeetcodeTask]:
    with open(FILE, 'r', encoding='utf-8') as file:
        table = file.read()

    # Split the content into lines
    lines = table.strip().split('\n')

    # Use pandas to read the table
    # Skip the first line (header) and the second line (separator)
    data = [line.split('|')[1:-1] for line in lines[2:]]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=[col.strip() for col in lines[0].split('|')[1:-1]])

    # Convert the DataFrame to a list of dictionaries
    list_of_objects = df.to_dict(orient='records')

    return [
        LeetcodeTask(
            obj["N"].strip(),
            obj["Problem"].strip(),
            obj["Difficulty"].strip(),
            obj["Count"].strip(),
            obj["Type"].strip(),
            obj["Comment"].strip(),
        )
        for obj in list_of_objects
    ]


tools = [select_random_leetcode_task]


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
        Your task is to select LeetCode tasks.
        You need to show the link, difficulty, and count.
        Do not show comments and types until I ask you.

        So, give me a random LeetCode task.
    """

    chat = ChatCohere()

    agent = LLMAgent(chat, tools)
    res = agent.invoke(content=system_prompt)

    while (True):
        print(res)
        res = agent.invoke(get_user_prompt())


if __name__ == '__main__':
    main()
