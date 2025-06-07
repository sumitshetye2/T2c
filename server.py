from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.requests import Request
from fastapi.exceptions import HTTPException
import os
import json
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import base64
import traceback

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#Gemini API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Please add your Gemini API key to the .env file.")

# Set the Gemini API key
genai.configure(api_key=gemini_api_key)

#Use Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/generate-problems")
def generate_problems(specification: str):
    try:
        # Decode the base64-encoded specification
        decoded_spec = base64.b64decode(specification).decode("utf-8")

        # Construct the system prompt
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

        # Call OpenAI API
        convo = model.start_chat()
        convo.send_message(system_prompt)
        response = convo.send_message(decoded_spec)

        ai_response = response.text.strip()

        if not ai_response or not response.text:
            raise ValueError("Empty response from Gemini")
        
        if ai_response.startswith("```json"):
            ai_response = ai_response.strip("```json").strip("```").strip()

        problems = json.loads(ai_response)

        return JSONResponse(content=problems)

    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating problems: {str(e)}")