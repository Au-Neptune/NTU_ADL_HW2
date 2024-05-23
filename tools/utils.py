import json

def convert_jsonl_to_data(jsonl_data):
    data = []
    with open(jsonl_data, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_data_to_jsonl(data, jsonl_data):
    with open(jsonl_data, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")