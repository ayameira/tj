import os
import json
import random
from openai import OpenAI
from datetime import datetime

OPENAI_API_KEY = "sk-bbmDxDlAMbtbL6THJu7mo0gTmGbgI7yGUOPtAWHdIAT3BlbkFJo-sBd6zx_q4TfX9WMDGNgMiUmRkWP9BW7NYLhY5NkA"
NUM_EXAMPLES = 0
MAX_VALIDATE = 100
timestmp=datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_PATH = f'{NUM_EXAMPLES}-SHOT_{timestmp}'
os.mkdir(RESULTS_PATH)
data_filepath = './data.json'


# ---------------- helpers ------------------ #
def write_to_disk(data, filename): 
    path = f'{RESULTS_PATH}/{filename}.json'
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def create_messages(examples):
    messages = []
    for sample in examples:
        messages.extend([
            {
                "role": "user",
                "content": [
                    {
                        "text": sample['dispositivo'],
                        "type": "text"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": json.dumps(sample['decisoes']),
                        "type": "text"
                    }
                ]
            }
        ])
    return messages

def get_completion(client, examples, dispositivo): 
    sys_message = [{
        "role": "system",
        "content": [
            {
                "text": "Você é um assistente judicial que identifica e classifica as decisões contidas em uma sentença judicial. Cada decisão diz respeito a um pedido do autor.",
                "type": "text"
            }
        ]
    }]
    example_messages = create_messages(examples)
    current_message = [{
        "role": "user",
        "content": [
            {
                "text": dispositivo,
                "type": "text"
            }
        ]
    }]
    messages = sys_message + example_messages + current_message

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "decisao_judicial",
                "schema": {
                    "type": "object",
                    "required": [
                    "decisoes"
                    ],
                    "properties": {
                    "decisoes": {
                        "type": "array",
                        "items": {
                        "type": "object",
                        "required": [
                            "pedido",
                            "tipoDaDecisao",
                            "valorDeferido",
                            "obrigacaoDeFazer"
                        ],
                        "properties": {
                            "pedido": {
                            "enum": [
                                "Dano moral",
                                "Dano material",
                                "Obrigação de fazer",
                                "Declaratória",
                                "Gratuidade da justiça"
                            ],
                            "type": "string",
                            "title": "Pedido"
                            },
                            "tipoDaDecisao": {
                            "enum": [
                                "Procedente",
                                "Improcedente",
                                "Parcialmente procedente",
                                "Não mencionado"
                            ],
                            "type": "string"
                            },
                            "valorDeferido": {
                            "type": "number"
                            },
                            "obrigacaoDeFazer": {
                            "type": "string",
                            "title": "Descrição da obrigação de fazer, se for o caso"
                            }
                        },
                        "additionalProperties": False
                        }
                    }
                    },
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    return response

def ChatCompletionObjectToDict(obj):
    return {
        'id': obj.id,
        'choices': [
            {
                'finish_reason': choice.finish_reason,
                'index': choice.index,
                'message_content': choice.message.content,
                'message_role': choice.message.role,
            }
            for choice in obj.choices
        ],
        'created': obj.created,
        'model': obj.model,
        'object': obj.object,
        'usage': {
            'completion_tokens': obj.usage.completion_tokens,
            'prompt_tokens': obj.usage.prompt_tokens,
            'total_tokens': obj.usage.total_tokens,
        }
    }

# ---------------   main   ------------------ #
client = OpenAI(
    api_key=OPENAI_API_KEY
)

# load data 
with open(data_filepath, 'r', encoding='utf-8') as file:
    data = json.load(file)

# divide data: examples (few shot) & validation
examples = []

for idx in range(NUM_EXAMPLES):
    rand_sample = random.choice(data)
    data.remove(rand_sample)
    examples.append(rand_sample)
    
while len(data) > MAX_VALIDATE:
    item = random.choice(data)
    data.remove(item)

# get predictions using api call
results = []
for sample in data:
    response = get_completion(client, examples, sample['dispositivo'])
    write_to_disk(ChatCompletionObjectToDict(response), sample['id'])
    decisoes = json.loads(response.choices[0].message.content)
    result = {
        "id": sample['id'],
        **decisoes
    }
    results.append(result)

write_to_disk(results, 'all_results')

# validate predictions
# log results (metrics?)


