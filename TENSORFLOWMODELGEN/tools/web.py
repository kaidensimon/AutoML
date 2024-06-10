import asyncio
import json
import sys
import aiohttp
from bs4 import BeautifulSoup
from langchain.tools import tool
from pydantic import BaseModel, Field


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
def parse_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in ["nav", "footer", "aside", "script", "style", "img", "header"]:
        for match in soup.find_all(tag):
            match.decompose()
    text_content = soup.get_text()
    text_content = " ".join(text_content.split())
    return text_content[:8_000]

async def get_webpage_content(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html_content = await response.text()
    text_content = parse_html(html_content)
    print(f"URL: {url} - fetched successfully")
    return text_content

class ResearchInput(BaseModel):
    research_urls: list[str] = Field(description="Must be a list of valid URLs.")

@tool("research", args_schema=ResearchInput)
async def research(research_urls: list[str]) -> str:
    """Get content of provided URLs for research purposes"""
    tasks = [asyncio.create_task(get_webpage_content(url)) for url in research_urls]
    #*tasks unpacks the list into seperate arguments
    contents = await asyncio.gather(*tasks, return_exceptions=True) #schedules multiple tasks, waits for all of them to finish
    return json.dumps(contents) #json dumps because we need to return all the content as string for the LLM



if __name__ == '__main__':

    import time

    TEST_URLS = [
        'https://en.wikipedia.org/wiki/Jordan_Belfort',
        'https://en.wikipedia.org/wiki/Oskar_Schindler',
        'https://en.wikipedia.org/wiki/Al_Capone'

    ]

    async def main():
        #ainvoke for asyncronous invoke
        result = await research.ainvoke({"research_urls": TEST_URLS})
        with open("test.json", "w") as f:
            json.dump(result, f)

    start_time = time.time()
    #asyncio.run creates a new event loop for every function its given
    #it also automatically closes the loop and returns the result of that loop for us
    asyncio.run(main())
    end_time = time.time()
    print(f"Async Time: {end_time - start_time} seconds")