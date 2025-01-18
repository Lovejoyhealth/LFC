import os
import asyncio
import threading
import json
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, models, CAR
from cerebras.cloud.sdk import Cerebras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI
app = FastAPI()

# Environment Variables
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-e2e8kypw838rwmpjxd9nx2vn5jrertm339fnrcnt9c6p8hmx")

# Initialize Cerebras Client
cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)

# In-memory storage for the latest relevant post
latest_post = None
lock = threading.Lock()

# Flag to control the Firehose client
client_running = True

class MarketPost(BaseModel):
    post: dict
    sentiment: str
    insights: str

def process_post(data: dict) -> Optional[dict]:
    """
    Process the post using Cerebras LLM to determine relevance and perform sentiment analysis.
    Returns the processed data if relevant, else None.
    """
    global cerebras_client

    document_text = data.get('text', '')

    # Define your prompts with explicit JSON instructions
    system_prompt = "You are a market sentiment analysis assistant."

    user_prompt = (
        "Please respond strictly with a JSON object containing the following fields:\n"
        "- 'relevant' (boolean): Indicates if the post is related to market sentiment.\n"
        "- 'sentiment' (string): The sentiment of the post ('positive', 'negative', 'neutral').\n"
        "- 'insights' (string): Market insights derived from the post.\n\n"
        "Do not include any additional text or explanations.\n\n"
        "Determine if the following post is related to market sentiment. "
        "If it is, analyze its sentiment and provide market insights. "
        "Exclude personal opinions and NSFW content.\n\n"
        f"Post: \"{document_text}\"\n\n"
        "Example Response:\n"
        "{\n"
        '  "relevant": true,\n'
        '  "sentiment": "positive",\n'
        '  "insights": "The user is offering graphic design services, indicating potential business collaboration opportunities."\n'
        "}"
    )

    try:
        completion = cerebras_client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=512,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )

        parsed_content = completion.choices[0].message.content

        # Attempt to parse the JSON response
        parsed_json = json.loads(parsed_content)

        # Validate the presence of required fields
        if all(key in parsed_json for key in ["relevant", "sentiment", "insights"]):
            if parsed_json["relevant"]:
                return {
                    "post": data,
                    "sentiment": parsed_json.get("sentiment"),
                    "insights": parsed_json.get("insights")
                }
        else:
            print(f"Missing keys in response: {parsed_json}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {parsed_content}")
    except Exception as e:
        print(f"Error processing post: {e}")

    return None

def on_message_handler(message):
    global latest_post, client_running

    commit = parse_subscribe_repos_message(message)
    if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
        return

    if not commit.blocks:
        return

    car = CAR.from_bytes(commit.blocks)
    for op in commit.ops:
        if op.action in ["create"] and op.cid:
            data = car.blocks.get(op.cid)

            if data and data.get('$type') == 'app.bsky.feed.post':
                processed = process_post(data)
                if processed:
                    with lock:
                        latest_post = processed
                    print("Processed and updated latest post.")
                if not client_running:
                    return

def run_firehose_client():
    global client_running
    client = FirehoseSubscribeReposClient()
    try:
        client.start(on_message_handler)
    except Exception as e:
        print(f"Firehose client stopped due to error: {e}")
    finally:
        client_running = False

# Start Firehose client in a separate thread
firehose_thread = threading.Thread(target=run_firehose_client, daemon=True)
firehose_thread.start()

@app.get("/latest-post", response_model=MarketPost)
async def get_latest_post():
    with lock:
        if latest_post:
            return latest_post
    raise HTTPException(status_code=404, detail="No relevant posts found yet.")