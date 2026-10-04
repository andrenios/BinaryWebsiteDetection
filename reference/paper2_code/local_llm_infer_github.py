# -*- coding: utf-8 -*-
import ollama
import uuid
import base64


def local_llm_infer(prompt,  max_tokens = 30, model = "gemma3:1b"):
    result = ollama.chat(
            model = model,
            messages = prompt,
            options={
                "stream": False,
                "num_predict": max_tokens,
                "temperature": 0.0,
                "think": False
    })
    return result['message']['content'].strip()

def local_llm_infer_v2(prompt_text, max_tokens=500, model="gemma3:1b"):
    
    result = ollama.generate(
        model=model,
        prompt=prompt_text,
        options={
            "num_predict": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 2107
        }
    )
    return result["response"].strip()

def local_llm_infer_v3(prompt_text, image_path = None, max_tokens=500, model="gemma3:1b"):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    result = ollama.generate(
        model=model,
        prompt=prompt_text,
        images=[image_data],
        options={
            "num_predict": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 2107
        }
    )
    return result["response"].strip()

