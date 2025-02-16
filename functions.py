import json
import os
import sqlite3
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
API_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"


client_local = OpenAI(api_key=AIPROXY_TOKEN, base_url=API_BASE_URL)


def install_and_run_script(script_url: str, email: str) -> List[Dict | str]:
    command = ["uv", "run", script_url, email]
    result = subprocess.run(
        command,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(command)
    return [
        {"response": "Task Done Successfully"},
        {"command": command},
        {"output": result},
    ]


def format_markdown_prettier(file_path: str, prettier_version: str):
    command = ["npx","-y", f"prettier@{prettier_version}", "--write", file_path]
    subprocess.run(command)
    return [{"response": "Task Done Successfully"}]


def count_days_and_save(day: str, input_file: str, output_file: str):
    # List of possible date formats (expanded)
    date_formats = [
        "%Y-%m-%d",  # 2025-02-13
        "%d-%b-%Y",  # 15-Sep-2002
        "%m/%d/%Y",  # 02/13/2025
        "%d/%m/%Y",  # 13/02/2025
        "%B %d, %Y",  # February 13, 2025
        "%Y/%m/%d %H:%M:%S",  # 2019/04/01 10:48:50
        "%Y-%m-%d %H:%M:%S",  # 2019-04-01 10:48:50
        "%d-%b-%Y %H:%M:%S",  # 15-Sep-2002 10:48:50
        "%b %d, %Y",  # Oct 31, 2001
    ]

    def parse_date(date_str: str):
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(f"Date '{date_str}' does not match any known formats.")

    with open(input_file, "r") as infile:
        dates = infile.readlines()

    day_count = sum(
        1 for date in dates if parse_date(date).strftime("%A").lower() == day.lower()
    )

    with open(output_file, "w") as outfile:
        outfile.write(str(day_count))

    return [{"response": "Task Done Successfully"}]


def sort_contacts(input_file: str, output_file: str, sort_keys: List[str]):
    with open(input_file, "r") as infile:
        contacts = json.load(infile)

    sorted_contacts = sorted(
        contacts, key=lambda x: tuple(x[key].lower() for key in sort_keys if key in x)
    )

    with open(output_file, "w") as outfile:
        json.dump(sorted_contacts, outfile, indent=4)

    return [{"response": "Task Done Successfully"}]


def generate_markdown_index(directory: str, output_file: str, tags: List[str]):
    index: Dict[str, str] = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        for tag in tags:
                            if line.startswith(tag + " "):
                                title = line[len(tag) :].strip()
                                index[relative_path] = title
                                break
                        if relative_path in index:
                            break

    with open(output_file, "w") as outfile:
        json.dump(index, outfile, indent=4)
    return [{"response": "Task Done Successfully"}]


def extract_using_llm(input_file: str, output_file: str, instructions: str):

    with open(input_file, "r") as file:
        content = file.read()

    response = client_local.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that extracts information based on user instructions.",
            },
            {"role": "user", "content": f"{instructions}\n\nText:\n{content}"},
        ],
    )

    extracted_info = response.choices[0].message.content.strip()

    with open(output_file, "w") as file:
        file.write(extracted_info)
    return [
        {
            "extracted_text_saved_to_location": output_file,
            "response": "Task Done Successfully.",
        }
    ]


def run_sql_query(
    database_file: str,
    query: str,
    output_file: Optional[str],
    database_type: str,
    output_format: str = "csv",
):

    if database_type == "sqlite":
        conn = sqlite3.connect(database_file)
    elif database_type == "duckdb":
        conn = duckdb.connect(database_file)
    else:
        raise ValueError("Unsupported database type. Use 'sqlite' or 'duckdb'.")

    try:
        result = pd.read_sql_query(query, conn)
    except Exception as e:
        conn.close()
        raise ValueError(f"Error executing query: {e}")
    finally:
        conn.close()

    # Save output if requested
    if output_file:
        if output_format == "csv":
            result.to_csv(output_file, index=False, header=True)  # CSV includes headers
        elif output_format == "txt":
            with open(output_file, "w") as file:
                for _, row in result.iterrows():
                    # Write rows without column names
                    file.write(" ".join(map(str, row.values)) + "\n")
        else:
            raise ValueError("Unsupported output format. Use 'csv' or 'txt'.")
    else:
        for _, row in result.iterrows():
            print(" ".join(map(str, row.values)))
            return {
                "output_saved_to_location": output_file,
                "response": "Task Done Successfully.",
            }

    return [{"response": result}]


def find_most_similar_texts(
    input_file: str, max_items: int = 1000, output_file: Optional[str] = None
) -> Tuple[str, str]:
    """
    Finds the most similar pair of text items in the given file using OpenAI embeddings.

    :param input_file: Path to the file containing the text items (one per line).
    :param max_items: Maximum number of items to process from the file (default: 1000).
    :param output_file: (Optional) Path to the file to write the most similar pair of items. If None, no file is written.
    :return: A tuple containing the most similar pair of items.
    """
    # Load the text items from the input file
    with open(input_file, "r", encoding="utf-8") as file:
        items = [line.strip() for line in file if line.strip()]

    if len(items) < 2:
        raise ValueError("The input file must contain at least two items to compare.")

    # Limit the number of items to process if specified
    items = items[:max_items]

    # Generate embeddings for all text items
    try:
        response = client_local.embeddings.create(
            input=items, model="text-embedding-3-small"
        )
        embeddings = np.array([data.embedding for data in response.data])
    except Exception as e:
        raise ValueError(f"Error while generating embeddings: {e}")

    # Compute dot product similarity between all pairs of embeddings
    similarity = np.dot(embeddings, embeddings.T)

    # Create mask to ignore diagonal (self-similarity)
    np.fill_diagonal(similarity, -np.inf)

    # Get indices of maximum similarity
    i, j = np.unravel_index(np.argmax(similarity), similarity.shape)

    most_similar_pair = (items[i], items[j])

    # Write the most similar pair to the output file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(most_similar_pair[0] + "\n")
            file.write(most_similar_pair[1] + "\n")

    return [
        {"output_saved_to_location": output_file, "response": "Task Done Successfully."}
    ]


def extract_text_from_image_using_llm(
    input_image: str, query: str, output_file: Optional[str] = None, image_format: str = "jpeg"
) -> str:
    """
    Extracts specific information from an image using an LLM based on the provided query.

    :param input_image: Path to the image file.
    :param query: Query specifying what to extract from the image.
    :param output_file: (Optional) Path to the file to write the extracted text. If None, no file is written.
    :param image_format: Format of the image (e.g., "jpeg", "png"). Defaults to "jpeg".
    :return: Extracted text as a string.
    """
    # Validate image format
    supported_formats = ["jpeg", "png", "webp"]
    if image_format.lower() not in supported_formats:
        raise ValueError(f"Unsupported image format: {image_format}. Supported formats are: {supported_formats}")

    # Convert the image to base64 (LLMs often accept images in base64 encoding for APIs)
    import base64

    with open(input_image, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Set the MIME type based on the image format
    mime_type = f"image/{image_format.lower()}"

    # Query the LLM
    try:
        response = client_local.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts information from images. Don't add any comments in the answer.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                            },
                        },
                    ],
                },
            ],
        )
        extracted_text = response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"Error while extracting text using LLM: {e}")

    # Save the extracted text to the output file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(extracted_text)
        return [
            {
                "output_saved_to_location": output_file,
                "response": "Task Done Successfully.",
            }
        ]

    return [{"response": extracted_text}]


def write_recent_logs(
    directory: str, output_file: str, num_files: int, num_lines: int, extension: str
):
    # Get all files with the specified extension in the directory
    files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(extension)
    ]

    # Get the most recent files based on modification time
    recent_files = sorted(files, key=lambda f: os.path.getmtime(f), reverse=True)[
        :num_files
    ]

    # Extract the specified number of lines from each file
    recent_logs = []
    for file_path in recent_files:
        with open(file_path, "r") as file:
            for _ in range(num_lines):
                line = file.readline().strip()
                if not line:  # Stop if there are no more lines
                    break
                recent_logs.append(line)

    # Write the recent logs to the output file
    with open(output_file, "w") as outfile:
        outfile.write("\n".join(recent_logs))
    return [
        {"output_saved_to_location": output_file, "response": "Task Done Successfully."}
    ]


def run_terminal_command(command: str):

    try:
        # Execute the shell command with subprocess.run for safety
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Return successful execution details
        return [{"status": "success", "output": result.stdout.strip(), "error": None}]
    except subprocess.CalledProcessError as e:
        # Return error details if command execution fails
        return [
            {
                "status": "error",
                "output": e.stdout.strip() if e.stdout else None,
                "error": e.stderr.strip() if e.stderr else str(e),
            }
        ]
    except Exception as e:
        # Handle other unexpected exceptions
        return [{"status": "error", "output": None, "error": str(e)}]
    

def commit_to_git_repo(commit_message: str, path_to_repo: str):
    """
    Commit changes to the specified Git repository.

    :param commit_message: Commit message to use.
    :param path_to_repo: Path to the local Git repository.
    """
    try:
        # Navigate to the repository directory
        if not os.path.exists(path_to_repo):
            return (f"The repository path '{path_to_repo}' does not exist.")
        
        # Check if it's a valid git repository
        if not os.path.exists(os.path.join(path_to_repo, ".git")):
            return (f"The path '{path_to_repo}' is not a valid git repository.")

        # Stage all changes
        subprocess.run(["git", "add", "."], cwd=path_to_repo, check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", commit_message], cwd=path_to_repo, check=True)

        return (f"Successfully committed changes with message: {commit_message}")
    except Exception as e:
        return (f"Error committing to repository: {e}")
    
def clone_git_repo(url_repo: str, output_dir: str):
    """
    Clone a Git repository to the specified directory.

    :param url_repo: URL of the repository to clone.
    :param output_dir: Local directory to clone the repository into.
    """
    try:
        subprocess.run(["git", "clone", url_repo, output_dir], check=True)
        return (f"Successfully cloned repository from {url_repo} to {output_dir}")
    except Exception as e:
        return (f"Error cloning repository: {e}")