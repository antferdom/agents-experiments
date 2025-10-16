import os
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY", "")
openai_api_base = "http://127.0.0.1:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

source_sentence = "The camera floats gently through rows of pastel-painted wooden beehives, buzzing honeybees gliding in and out of frame. The motion settles on the refined farmer standing at the center, his pristine white beekeeping suit gleaming in the golden afternoon light. He lifts a jar of honey, tilting it slightly to catch the light. Behind him, tall sunflowers sway rhythmically in the breeze, their petals glowing in the warm sunlight. The camera tilts upward to reveal a retro farmhouse with mint-green shutters, its walls dappled with shadows from swaying trees. Shot with a 35mm lens on Kodak Portra 400 film, the golden light creates rich textures on the farmer’s gloves, marmalade jar, and weathered wood of the beehives."
prompt_template = "Can you translate form English to Chinese (simplified) the following prompt: {}  /think"

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-250425",
    messages=[
        {"role": "user", "content": prompt_template.format(source_sentence)},
    ],
    max_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
        "separate_reasoning": True
    }, 
)

print("==== Reasoning ====")
print(chat_response.choices[0].message.reasoning_content)

print("==== Text ====")
print(chat_response.choices[0].message.content)