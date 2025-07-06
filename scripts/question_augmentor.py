import json
from openai import OpenAI
import random
import time
from tqdm import tqdm
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def paraphrase_question(question, n=3):
    prompt = f"Paraphrase the following question in {n} different natural ways:\n\nQ: {question}\n\nParaphrased versions:"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.7,
        max_tokens=200
    )

    content = response.choices[0].message.content.strip()
    # Assume output is formatted as a list
    lines = content.split('\n')
    cleaned = [line.lstrip("-•1234567890. ").strip() for line in lines if line.strip()]
    return cleaned[:n]


def augment_questions(qa_path, out_path):
    with open(qa_path) as f:
        data = json.load(f)

    new_data = []
    for item in tqdm(data):
        new_qas = []
        for qa in item["qas"]:
            try:
                variations = paraphrase_question(qa["question"], n=3)
                for q in variations:
                    new_qas.append({
                        "question": q,
                        "answer": qa["answer"],
                        "bbox": qa["bbox"]
                    })
                time.sleep(1.2)  # avoid rate limits
            except Exception as e:
                print(f"Error: {e}")
                new_qas.append(qa)
        item["qas"] = new_qas
        new_data.append(item)

    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)


if __name__ == "__main__":
    augment_questions(
        qa_path="../data/training/generated_qa.json",
        out_path="../data/training/generated_qa_augmented.json"
    )
