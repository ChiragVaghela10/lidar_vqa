# Generate generated_qa.json for structured QA pairs
import os
import json
import random
from tqdm import tqdm

DATA_DIR = "../data/training"
LABEL_DIR = os.path.join(DATA_DIR, "label_2")
IMAGE_DIR = os.path.join(DATA_DIR, "image_2")

def parse_label_line(line):
    fields = line.strip().split()
    obj_class = fields[0]
    bbox = list(map(float, fields[4:8]))  # [x1, y1, x2, y2]
    return obj_class, bbox

def generate_qa_entry(image_id, objects):
    qas = []
    for obj in objects:
        cls, bbox = obj["class"], obj["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        templates = {
            "Car": [
                f"What is the vehicle in the region ({x1}, {y1}, {x2}, {y2})?",
                f"Is there a car in this image?",
                f"What object is in front of the camera?",
                f"What kind of vehicle is detected?"
            ],
            "Pedestrian": [
                f"Who is walking in the region ({x1}, {y1}, {x2}, {y2})?",
                f"Is there a person on the road?",
                f"Who is visible near the sidewalk?",
                f"Who is crossing the street?"
            ],
            "Cyclist": [
                f"Who is riding in this scene?",
                f"Is there a cyclist in the image?",
                f"What moving object is near the curb?",
                f"Who is on two wheels in this view?"
            ]
        }

        question = random.choice(templates.get(cls, [
            f"What is in the region ({x1}, {y1}, {x2}, {y2})?"
        ]))

        qas.append({
            "question": question,
            "answer": f"A {cls.lower()}",
            "bbox": bbox
        })
    return {
        "image_id": image_id,
        "image_path": f"image_2/{image_id}.png",
        "qas": qas
    }

def main():
    output = []
    for fname in tqdm(sorted(os.listdir(LABEL_DIR))):
        image_id = fname.replace(".txt", "")
        with open(os.path.join(LABEL_DIR, fname)) as f:
            objects = []
            for line in f:
                cls, bbox = parse_label_line(line)
                if cls in ["Car", "Pedestrian", "Cyclist"]:  # filter useful classes
                    objects.append({"class": cls, "bbox": bbox})
            if objects:
                entry = generate_qa_entry(image_id, objects)
                output.append(entry)

    with open(os.path.join(DATA_DIR, "generated_qa.json"), "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()

