import os

from decouple import config
from datetime import date

def set_environment_variables(project_name : str = "") -> None:
    if not project_name:
        project_name = f"Test_{date.today()}"
    os.environ["OPENAI_API_KEY"] = str(config("",
                                              default=""))

    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    os.environ["LANGCHAIN_API_KEY"] = str(config("",
                                                 default=""))
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ['TAVILY_API_KEY'] = str(
        config("", default=""))

    print("API KEYS LOAED AND TRACING SET WITH PROJECT NAME", project_name)


