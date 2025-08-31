import asyncio
from agents import Agent, Runner,enable_verbose_stdout_logging, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, function_tool
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

gemini_model = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=gemini_model
)

@function_tool
def calculate_area(width, length):
    """calculate area when user calls tools"""
    area = width * length
    return f"width {width} * length {length} = {area}"
    
# 1) Specialists
summer = Agent(
    name="Summarizer",
    model=model,
    instructions="Summarize in 3 bullet points. No extra text."
)
detect_lang = Agent(
    name="Language Detector",
    model=model,
    instructions="Return the ISO language code for the given text."
)

# 2) Wrap as tools (quick pattern)
summarize_tool = summer.as_tool("summarize", "Summarize in 3 bullets.")
detect_tool = detect_lang.as_tool("detect_language", "Detect language code.")

main_agent = Agent(
    name="agent",
    instructions="you are a agent -can use tools as user queries",
    model=model,
    tools = [calculate_area, summarize_tool,detect_tool],
)

config:RunConfig = RunConfig(
    set_tracing_disabled(disabled=True),
    enable_verbose_stdout_logging(),
)

input_prompt = "summarize The morning air was crisp and cool, carrying with it the faint scent of rain from the night before. Birds fluttered between the branches, their songs weaving together like an unplanned symphony."

async def agent_run() : 
    run = await Runner.run(main_agent, input = input_prompt,run_config=config)
    print(run)
    
asyncio.run(agent_run())