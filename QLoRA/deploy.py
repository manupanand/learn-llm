#deploy.py
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
import string
import random

# initializing size of string
N = 20

# using random.choices()
# generating random strings
res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))

generator = OpenAIChatGenerator(
    api_key=Secret.from_token(res),  # for compatibility with the OpenAI API, a placeholder api_key is needed
    model="/content/final_weights_new",
    api_base_url="http://localhost:8000/v1",
    generation_kwargs = {"max_tokens": 1024, "temperature":0.01}
)

messages = []

while True:
  msg = input("Enter your message or Q to exit\n🧑 ")
  if msg=="Q":
    break
  messages.append(ChatMessage.from_user(msg))
  response = generator.run(messages=messages)
  #print(response)
  assistant_resp = response['replies'][0]
  print(assistant_resp.text) # Access and print the text content