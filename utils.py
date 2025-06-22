from typing import TypeVar, List, Dict, Any
import pandas as pd

T = TypeVar('T')


def convert_file_to_object_list(file_path: str, constructor: callable) -> List[T]:
    with open(file_path, 'r', encoding='utf-8') as file:
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
        constructor(
            obj["N"].strip(),
            obj["Problem"].strip(),
            obj["Difficulty"].strip(),
            obj["Count"].strip(),
            obj["Type"].strip(),
            obj["Comment"].strip(),
        )
        for obj in list_of_objects
    ]


def convert_object_list_to_file(task_list: List[T], file_path: str, attributes: List[str]) -> None:
    # Create the header and separator
    header = "| " + " | ".join(["N", "Problem", "Difficulty", "Count", "Type", "Comment"]) + " |"
    separator = "|---" + "|---" * (len(attributes) - 1) + "|"

    # Prepare the data rows
    data_rows = []
    for task in task_list:
        data_rows.append(
            "| " + " | ".join(str(getattr(task, attr)) for attr in attributes) + " |"
        )

    # Combine header, separator, and data rows into a single string
    content = "\n".join([header, separator] + data_rows)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
