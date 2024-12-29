import json_repair
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

app = FastAPI()
hf_token = os.getenv("HF_TOKEN", "token")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
    token=hf_token
)

#distil = FastLanguageModel.from_pretrained(
#    model_name="irlab-udc/Llama-3-8B-Distil-MetaHate",
#    max_seq_length=4096,
#    dtype=None,
#    load_in_4bit=True,
#    token=hf_token
#)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

FastLanguageModel.for_inference(model)
#FastLanguageModel.for_inference(distil)

hs_instruction = """
You must explain why a social media message is hate or not and then tell me your decision. You must always reply with only a JSON containing one field 'hate_speech' including a Boolean value ("True" for hate speech messages, "False" for neutral ones); and a field 'explanations' containing a list with the each message phrase and its corresponding explanation. Do not include text outside the JSON.
This is the definition of hate speech: "language characterized by offensive, derogatory, humiliating, or insulting discourse that promotes violence, discrimination, or hostility towards individuals or groups based on attributes such as race, religion, ethnicity, or gender".

The input format is:
    Generate step-by-step explanation for:\n<Message><input query></Message>.
The output format is:
    {
        "hate_speech": "<Boolean>",
        "explanations": [
            {
                "input": "<input query phrase 1>",
                "explanation": "<input query 1 phrase step-by-step explanation>"
            },
            {
                "input": "<input query phrase 2>",
                "explanation": "<input query 2 phrase step-by-step explanation>"
            }
        ]
    }
Generate step-by-step explanation for:"""
hyperpartisan_instruction = """###Instruction: 
You must determine if an article is hyperpartisan by following the reasoning steps below. You must always reply with only a JSON containing one field 'hyperpatisan' including a Boolean value ("True" for hyperpartisan messages, "False" for neutral ones); and a field 'explanations' containing a list with the each reasoning step and its corresponding explanation. Do not include text outside the JSON.
Reason step by step as follows: 
1. Sentiment analysis ('sentiment_analysis'): Analyze the tone and language to see if there are polarizing words or emotional language.
2. Rhetorical bias ('rhetorical_bias'): Rhetoric refers to speaking or writing designed to have a persuasive or impressive effect but lacking meaningful content. Analyze the presence of rhetorical biases like ad hominem attacks.
3. Framing bias ('framing_bias'): Assess how the information is presented to shape or influence perceptions by emphasizing certain aspects while downplaying others.
4. Ideological bias ('ideological_bias'): Determine if specific moral values linked to a particular ideology appear when carefully reading the text.
5. Intention ('intention'): Analyze the intent of the article. Does it aim to persuade or merely inform?
6. Unilateral coverage ('unilateral_coverage'): Does the article provide only one point of view? Is it unilateral in its coverage?
7. Left-wing hyperpartisan ('left_wing_hyperpartisan'): Consider yourself a left-wing reader. Would you consider this article hyperpartisan from your political stance? [Follow the instructions from 1 to 6.]
8. Right-wing hyperpartisan ('right_wing_hyperpartisan'): Consider yourself a right-wing reader. Would you consider this article hyperpartisan from your political stance? [Follow the instructions from 1 to 6.]
9. Hyperpatisan ('hyperpatisan'): "True" for hyperpartisan messages, "False" for neutral ones. [Follow the instructions skipping steps 7 and 8.] 

The input format is:
    Generate step-by-step explanation for:\n<Message><input query></Message>.
The output format is:
    {
        "hyperpatisan": "<Boolean>",
        "explanations": [
            {
                "sentiment_analysis": "<Sentiment analysis explanation>"
            },
            {
                "rhetorical_bias": "<Rhetorical bias explanation>"
            },
            {
                "framing_bias": "<Framing bias explanation>"
            },
            {
                "ideological_bias": "<Ideological bias explanation>"
            },
            {
                "intention": "<Intention explanation>"
            },
            {
                "unilateral_coverage": "<Unilateral coverage explanation>"
            },
            {
                "left_wing_hyperpartisan": "<Left wing hyperpartisan explanation>"
            },
            {
                "right_wing_hyperpartisan": "<Right wing hyperpartisan explanation>"
            },
        ]
    }
Generate step-by-step explanation for:
"""
fake_news_instruction = """
You must detect if an article is is fake or not. You must always reply with only a JSON containing one field 'fake_news' including a Boolean value ("True" for fake news messages, "False" for real ones); and a field 'explanations' containing a list with the each reasoning step and its corresponding explanation. Do not include text outside the JSON.
Reason step by step as follows: 
1. Sentiment analysis ('sentiment_analysis'): Analyze the tone and language to see if there is unintentional or intentional harmful behavior, words, or emotional language.
2. Identify the target audience ('target'): Who seems to be the target audience for this article?
3. Disinformation and false information ('disinformation'): The spreading of false or misleading information, often with the intention to mislead or manipulate. This can include conspiracy theories, fake news or manipulated media.
9. Fake news ('fake_news'): "True" for fake news messages, "False" for real ones. [Follow the instructions.] 

The input format is:
    Generate step-by-step explanation for:\n<Message><input query></Message>.
The output format is:
    {
        "fake_news": "<Boolean>",
        "explanations": [
            {
                "sentiment_analysis": "<Sentiment analysis explanation>"
            },
            {
                "target": "<Target explanation>"
            },
            {
                "disinformation": "<Disinformation explanation>"
            }
        ]
    }
Generate step-by-step explanation for:
"""


class PromptRequest(BaseModel):
    id: str
    prompt: str


def inference(instruction: str, prompt: str, is_distil=False):
    message = [{"from": "human", "value": instruction + "<Message>" + prompt + "</Message>"}]
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    #if is_distil:
    #    outputs = distil.generate(input_ids=inputs, max_new_tokens=2048, use_cache=True)
    #else:
    outputs = model.generate(input_ids=inputs, max_new_tokens=2048, use_cache=True)
    result = tokenizer.batch_decode(outputs)
    decoded_object = json_repair.repair_json(result[0], return_objects=True)
    return decoded_object[-1]


def generate_user_messages(hate_speech_response, fake_news_response, hyperpartisan_response):
    """
    Generates a user-friendly message string for each detected harm based on model responses.

    Parameters:
        hate_speech_response (dict): Response from the hate speech model.
        fake_news_response (dict): Response from the fake news model.
        hyperpartisan_response (dict): Response from the hyperpartisan model.

    Returns:
        str: A single string containing all messages to display to the user.
    """
    messages = []

    # Process hate speech response
    if hate_speech_response.get("hate_speech", False):
        explanation = "; ".join(
            f"{entry['input']}: {entry['explanation']}"
            for entry in hate_speech_response.get("explanations", [])
        )
        messages.append(f"This content contains hate speech. {explanation}.")

    # Process fake news response
    if fake_news_response.get("fake_news", False):
        explanation = "; ".join(
            f"{key}: {value}"
            for entry in fake_news_response.get("explanations", [])
            for key, value in entry.items()
        )
        messages.append(f"This content contains fake news. {explanation}.")

    # Process hyperpartisan news response
    if hyperpartisan_response.get("hyperpatisan", False):
        explanation = "; ".join(
            f"{key}: {value}"
            for entry in hyperpartisan_response.get("explanations", [])
            for key, value in entry.items()
        )
        messages.append(f"This content contains hyperpartisan news. {explanation}.")

    # Add a message if no harms were detected
    if not messages:
        messages.append("No online harms detected in this content.")

    # Combine messages into a single string
    return "\n".join(messages)


@app.get("/")
def root():
    return {"message": "Model API is running on GPU with 4-bit quantized Llama!"}


@app.post("/analyze/")
async def generate_text(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        # Run inference for all models concurrently in threads
        hs_response = inference(hs_instruction, request.prompt)
        hyperpartisan_response = inference(hyperpartisan_instruction, request.prompt)
        fake_news_response = inference(fake_news_instruction, request.prompt)
        response = generate_user_messages(hs_response, fake_news_response, hyperpartisan_response)

        return {
            "id": request.id,
            "text": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
