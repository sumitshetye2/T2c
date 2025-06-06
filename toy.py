#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["google-generativeai", "python-dotenv"]
# ///

import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Gemini API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Please add your Gemini API key to the .env file.")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

# Create the Gemini model instance
model = genai.GenerativeModel("gemini-2.0-flash")  # Or another version if desired

def generate_problems(task_specification):
    """
    Generate programming problems based on a task specification.

    Args:
        task_specification (dict): A dictionary containing the task specification.

    Returns:
        Generator[str]: Yields text chunks of the generated response.
    """
    system_prompt = """
You are a generator of Parsons-style programming puzzles.

Each puzzle consists of shuffled code blocks. Your job is to produce a list of such puzzles in JSON format. Each puzzle must help learners practice constructing correct logic from a mix of relevant and misleading code blocks.

Use the following JSON structure for each problem:

[
  {
    "title": "A short description of the task",
    "codeBlocks": [
      "... all code blocks (correct ones + distractors)"
    ],
    "correctOrder": [
      "... only the blocks needed, in the correct order"
    ],
    "distractors": [
      "... blocks from codeBlocks that are not part of the correct solution"
    ],
    "hint": "A tip to help students figure out the correct structure"
  },
  ...
]

Guidelines:
- `codeBlocks` must include all lines needed to solve the problem, plus distractors.
- `correctOrder` must be a subset of `codeBlocks` arranged in correct logical order.
- `distractors` must be a subset of `codeBlocks` that are not in `correctOrder`.
- The hint should be helpful but not give away the full solution.
- Avoid trivial distractors. Aim for plausible-looking alternatives.
- Use only concepts and difficulty levels provided in the task specification.
- Do not output any text before or after the JSON.
- The final output must be a **JSON array of problem objects** as shown above.
"""



    user_message = system_prompt + "\n\nTask Specification:\n" + json.dumps(task_specification)

    response = model.generate_content(
        user_message,
        stream=True,
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text


if __name__ == "__main__":
    # Example task specification
    task_spec = {
        "language": "Python",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": True},
            "Medium": {"Functions": True},
            "Hard": {"Recursion": False},
        },
        "num_problems": 2,
    }

    # Generate problems
    deltas = generate_problems(task_spec)

    # Print the streaming output
    for delta in deltas:
        print(delta, end="")
