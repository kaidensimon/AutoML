from pydantic import BaseModel, Field
import os
from langchain.tools import tool


class DirInput(BaseModel):
    input_dir: str = Field(
        description="Must be a valid path to your data directory"
    )


@tool("inspect_data_dir", args_schema=DirInput)
def inspect_data_dir(input_dir: str) -> str:
    """Inspect the data directory to see number of class folders to determine what loss function you should use."""
    if not input_dir:
        return (
            "Please provide a valid data directory"
        )
    classes = []

    for i in os.listdir(input_dir):
        classes.append(i)

    num_classes = len(classes)

    if num_classes > 2:
        return f"You are doing multi-class classification. The number of classes is {num_classes}"
    else:
        return "You are doing binary classification. The number of classes is 1."


if __name__ == "__main__":
    print(inspect_data_dir.run(r"D:\fastaimodelmaker\Data"))