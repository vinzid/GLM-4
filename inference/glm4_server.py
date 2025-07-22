import sys
from threading import Thread
from typing import List, Literal, Optional

import torch
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


app = FastAPI()


class MessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None


class MessageOutput(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class Choice(BaseModel):
    message: MessageOutput


class Request(BaseModel):
    messages: List[MessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 128000
    repetition_penalty: Optional[float] = 1.0


class Response(BaseModel):
    model: str
    choices: List[Choice]


@app.post("/v1/chat/completions", response_model=Response)
async def create_chat_completion(request: Request):
    global model, tokenizer

    print(datetime.now())
    print("\033[91m--received_request\033[0m", request)
    messages = [message.model_dump() for message in request.messages]
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "streamer": streamer,
        "max_new_tokens": request.max_tokens,
        "do_sample": True,
        "top_p": request.top_p,
        "temperature": request.temperature if request.temperature > 0 else 0.8,
        "repetition_penalty": request.repetition_penalty,
        "eos_token_id": model.config.eos_token_id,
    }
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    result = ""
    for new_token in streamer:
        result += new_token
    print(datetime.now())
    print("\033[91m--generated_text\033[0m", result)

    message = MessageOutput(
        role="assistant",
        content=result,
    )
    choice = Choice(
        message=message,
    )
    response = Response(model=sys.argv[1].split("/")[-1].lower(), choices=[choice])
    return response


torch.cuda.empty_cache()

if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
