from pydantic import Field, BaseModel
from langchain.tools import tool
import subprocess
class RunDirInput(BaseModel):
    input_script: str = Field(
        description="Must be a valid directory to a python file."
    )

@tool("run_model", args_schema=RunDirInput)
def run_model(input_script: str) -> str:
    """Runs the machine learning model script"""
    if not input_script:
        return(
            "Please input a valid path to a python script"
        )
    subprocess.run(['python', input_script])


if __name__ == "__main__":
    print(run_model.run())