from pydantic import Field, BaseModel
from langchain.tools import tool
import os
class PyFileInput(BaseModel):
    script: str = Field(
        description="Must be a valid string with correct python syntax formatting"
    )

@tool("generate_python_script", args_schema=PyFileInput)
def generate_python_script(script: str) -> str:
    """Generate a python script with the code you generate from the internet sources provided"""
    if not script:
        return(
            "Please provide a string"
        )
    filepath = os.getcwd()
    file_name = "script.py"
    temp_path = filepath + file_name
    with open(file_name, 'w') as f:
        f.write(script)
    print('execution complete')