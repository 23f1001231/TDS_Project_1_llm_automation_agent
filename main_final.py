import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from function_definations import tools
from process_task import execute_function, system_prompt

load_dotenv(".env")

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
API_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

os.environ["OPENAI_API_BASE"] = "https://aiproxy.sanand.workers.dev/openai/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("AIPROXY_TOKEN")

client = OpenAI(api_key=AIPROXY_TOKEN, base_url=API_BASE_URL)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"hello": "world"}


@app.get("/read")
async def read_file(path: str):
    # Ensure safe path resolution
    base_dir = Path("/data").resolve()  # Base directory
    file_path = (base_dir / path).resolve()  # Resolve to an absolute path

    # Check if the resolved path is still within the base directory
    if not file_path.is_file() or base_dir not in file_path.parents:
        return Response(status_code=404)

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.post("/run")
async def handle_task(task: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    limit = 0
    while limit < 4:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        response_message = response.choices[0].message
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print("--------")
                print(name, args)
                print("--------")
                function_response = execute_function(name, args)
                print("--------")
                print(function_response)
                print("--------")
                messages.append({"role": "system", "content": str(function_response)})
            limit = limit + 1
        else:
            return {"result": response_message.content}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
