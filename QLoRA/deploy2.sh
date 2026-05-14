# =============================================================================
# deploy.py — Interactive Chat Client via Haystack + vLLM OpenAI API
# This script talks to your vLLM server (started in dependency.sh)
# using the Haystack framework as the client interface.
# =============================================================================

# =============================================================================
# BLOCK 1: IMPORTS
# OpenAIChatGenerator: Haystack component that calls any OpenAI-compatible API
#   - vLLM exposes OpenAI-compatible endpoints — so this works with your local model
#   - You don't need an actual OpenAI account or real API key
# ChatMessage: Haystack's message object — wraps role (user/assistant) + content
# Secret: Haystack's way of handling API keys securely
# =============================================================================

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
import string
import random
import os

# =============================================================================
# BLOCK 2: PROMPT TEMPLATE (must match fine-tune.py exactly)
# The model was trained on this exact format — any deviation causes bad output
# =============================================================================

input_prompt = """Below is a Human Input, write appropriate Response based on the input.

### Input:
{}

### Response:
{}"""

# =============================================================================
# BLOCK 3: FAKE API KEY GENERATOR
# vLLM requires an api_key field for OpenAI API compatibility
# but it doesn't actually validate the key — it's just a placeholder
# The original code generates a random string for this — correct approach
# CORRECTED: made the intent clearer with a comment
# =============================================================================

# Generate a random placeholder API key (vLLM doesn't validate it)
fake_api_key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))

# =============================================================================
# BLOCK 4: INITIALIZE THE GENERATOR
# OpenAIChatGenerator connects to your vLLM server at localhost:8000
# model: must match the --model path you passed to vLLM server
# api_base_url: points to your local vLLM server, not OpenAI's servers
# generation_kwargs:
#   - max_tokens: maximum tokens the model generates per response
#   - temperature: controls randomness. 0.0 = deterministic, 1.0 = creative
#     For a knowledge bot, keep low (0.01-0.1) — you want consistent answers
# =============================================================================

generator = OpenAIChatGenerator(
    api_key=Secret.from_token(fake_api_key),
    model="./final_weights_new",        # Must match vLLM --model argument
    api_base_url="http://localhost:8000/v1",
    generation_kwargs={
        "max_tokens": 1024,
        "temperature": 0.1,             # CORRECTED: 0.01 → 0.1, 0.01 causes issues with some samplers
    }
)

# =============================================================================
# BLOCK 5: FORMAT USER MESSAGE IN PROMPT TEMPLATE
# ADDED: This was missing in the original deploy.py
# The model expects input in the exact prompt format it was trained on
# Without this, the model doesn't know it's in "Q&A mode" and gives poor responses
# =============================================================================

def format_user_message(user_input: str) -> str:
    """Wrap user input in the training prompt template."""
    return input_prompt.format(user_input, "")  # Leave Response blank — model fills it

def extract_response(full_text: str) -> str:
    """Extract only the model's response after ### Response:"""
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[1].strip()
        # Stop at next ### if model generates extra sections
        response = response.split("###")[0].strip()
        return response
    return full_text.strip()

# =============================================================================
# BLOCK 6: CONVERSATION LOOP
# messages: list of ChatMessage objects — full conversation history
# This is sent to the model every turn so it has context of prior messages
# Why maintain history? The model is stateless — it has no memory.
#   Sending full history simulates multi-turn conversation.
# "Q" to quit — simple exit condition
# =============================================================================

print("Chat started. Type 'Q' to exit.\n")
messages = []

while True:
    # Get user input
    msg = input("🧑 You: ")

    if msg.strip().upper() == "Q":
        print("Exiting chat.")
        break

    if not msg.strip():
        continue

    # ADDED: Format the message in the training prompt template
    formatted_msg = format_user_message(msg)

    # Append user message to conversation history
    messages.append(ChatMessage.from_user(formatted_msg))

    # Send full conversation history to model
    try:
        response = generator.run(messages=messages)

        # Extract the assistant's reply from the response object
        # response['replies'] is a list of ChatMessage objects
        assistant_message = response['replies'][0]
        response_text = assistant_message.text

        # ADDED: Clean up the response to show only the answer
        clean_response = extract_response(response_text)

        print(f"\n🤖 Bot: {clean_response}\n")

        # Append assistant response to history for multi-turn context
        # CORRECTED: original didn't append assistant response — broke multi-turn
        messages.append(ChatMessage.from_assistant(clean_response))

    except Exception as e:
        print(f"Error: {e}")
        print("Is the vLLM server running? Check: tail -f vllm.log")
        break
