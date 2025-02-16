from typing import List, Optional

from openai import pydantic_function_tool
from pydantic import BaseModel, Field


class InstallAndRunDatagen(BaseModel):
    """Model for installing dependencies (if required) and running a Python script."""

    script_url: str = Field(
        ..., description="URL to the Python script to be downloaded and executed."
    )

    email: str = Field(..., description="The email that is passed as the argument")


class CountTheNumberofDaysAndSave(BaseModel):
    day: str = Field(..., description="Name of the day")
    input_file: str = Field(
        ..., description="Path to the file containing a list of dates."
    )
    output_file: str = Field(
        ..., description="Path to the file to write the count of the days."
    )


class SortContacts(BaseModel):
    input_file: str = Field(
        ..., description="Path to the JSON file containing contacts."
    )
    output_file: str = Field(
        ..., description="Path to the JSON file to write the sorted contacts."
    )
    sort_keys: List[str] = Field(
        ..., description="List of keys to sort by, in order of priority."
    )


class WriteRecentLogs(BaseModel):
    directory: str = Field(
        ..., description="Path to the directory containing log files."
    )
    output_file: str = Field(
        ..., description="Path to the file to write the recent logs."
    )
    num_files: int = Field(..., description="Number of recent files to process.")
    num_lines: int = Field(
        ..., description="Number of lines to extract from each file."
    )
    extension: str = Field(
        ..., description="File extension to filter by, e.g., '.log'."
    )


class GenerateMarkdownIndex(BaseModel):
    directory: str = Field(
        ..., description="Path to the directory containing Markdown files."
    )
    output_file: str = Field(
        ..., description="Path to the JSON file to write the index."
    )
    tags: List[str] = Field(
        ..., description="List of Markdown tags to extract, e.g., ['#', '##']."
    )


class ExtractUsingLLM(BaseModel):
    input_file: str = Field(
        ..., description="Path to the file containing the text to be processed."
    )
    output_file: str = Field(
        ..., description="Path to the file to write the extracted information."
    )
    instructions: str = Field(
        ...,
        description="""Instructions for the LLM to extract the desired information.
                              Follow the Instruction exactlty. Example: If it says to JUST extract the email: Remember to prompt the tool to just output the email and nothing else.
                              """,
    )


class ExtractTextFromImageUsingLLM(BaseModel):
    """
    Model for extracting specific information from an image using an LLM.
    """

    input_image: str = Field(..., description="Path to the image file.")
    query: str = Field(
        ...,
        description="""Query specifying what to extract from the image . Specify with or without spaces.).
                       Remember to focus on important details.
                       """,
    )
    output_file: Optional[str] = Field(
        ...,
        description="Path to the file to write the extracted text. If None, no file will be written.",
    )
    image_format: str = Field(
        ...,
        description='Format of the image (e.g., "jpeg", "png"). Defaults to "jpeg"'
    )


class FindMostSimilarTextsUsingEmbeddings(BaseModel):
    """
    Pydantic model for the `find_most_similar_texts` function.
    """

    input_file: str = Field(
        ..., description="Path to the file containing the text items, one per line."
    )
    max_items: Optional[int] = Field(
        ..., description="Maximum number of items to process from the file."
    )
    output_file: Optional[str] = Field(
        ...,
        description="Path to the file to write the most similar pair. If None, output is not written to a file.",
    )


class CalculateTotalTicketSales(BaseModel):
    database_file: str = Field(..., description="Path to the SQLite database file.")
    output_file: str = Field(
        ..., description="Path to the file to write the total sales."
    )
    ticket_type: str = Field(
        ..., description="The ticket type to calculate total sales for."
    )


class RunSQLQuery(BaseModel):
    database_file: str = Field(
        ..., description="Path to the SQLite or DuckDB database file."
    )
    query: str = Field(..., description="The SQL query to be executed.")
    output_file: Optional[str] = Field(
        ...,
        description="Path to the file to save the query output. If None, no output will be saved.",
    )
    database_type: str = Field(
        ..., description="The type of database. Either 'sqlite' or 'duckdb'."
    )
    output_format: Optional[str] = Field(
        ...,
        description="The format of the output file. Either 'csv' or 'txt'. Defaults to 'csv'.",
    )


class Execute_shell_command(BaseModel):
    command: str = Field(..., description="Shell commnands to execute in the terminal")

class CommitMessageToGitRepo(BaseModel):
    commit_message: str = Field(..., description="Message to commit in git repo. Cannot be empty. If no message is provided, default message is '.' ")
    path_to_repo: str = Field(..., description="{Path to the repository folder}")

class Git_Clone(BaseModel):
    url_repo: str = Field(..., description="URL of the git repository to clone")
    output_dir: str = Field(..., description="The directory where the repository will be cloned.")
tools = [
    pydantic_function_tool(CountTheNumberofDaysAndSave),
    pydantic_function_tool(SortContacts),
    pydantic_function_tool(WriteRecentLogs),
    pydantic_function_tool(GenerateMarkdownIndex),
    pydantic_function_tool(ExtractUsingLLM),
    pydantic_function_tool(ExtractTextFromImageUsingLLM),
    pydantic_function_tool(FindMostSimilarTextsUsingEmbeddings),
    pydantic_function_tool(RunSQLQuery),
    pydantic_function_tool(Git_Clone),
    pydantic_function_tool(CommitMessageToGitRepo),
    # Very dangerous
    pydantic_function_tool(Execute_shell_command),
]
