from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(1)
import re 
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import os

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

MODEL_PATH = os.getenv("MODEL_PATH", "./saved_summary_model")

# If MODEL_PATH is a full URL, strip the huggingface prefix to extract the repository ID
if MODEL_PATH.startswith("https://huggingface.co/"):
    MODEL_PATH = MODEL_PATH.replace("https://huggingface.co/", "")

# Check if config.json exists locally; if not, fallback to Hugging Face repository
if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")) and MODEL_PATH == "./saved_summary_model":
    print("Warning: Local model config not found. Falling back to Hugging Face repository 'Senjuti3/my-t5-summarizer'.")
    MODEL_PATH = "Senjuti3/my-t5-summarizer"

model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    # decoded our output
    summary = tokenizer.decode(targets[0].cpu(), skip_special_tokens=True) # EOS, SEP
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})/
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )