# ADL HW2

## Project Description
This project is part of the NTU Applied Deep Learning course (Fall 2023) and focuses on text summarization task using the mT5 model from Google. The goal of the project is to train the mT5 model to generate summaries of given texts.

The project provides a script for setting up the required environment, a script for downloading necessary models, tokenizers, and data, and a script for running the trained model to predict results.

![Task Description](./images/Task%20Description.png)

for more information please refer to [ADL2023-HW2](./ADL2023-HW2.pdf)

## Enviroments

```bash
pip install -r requirements.txt
```

## Quick Start

download the zip of models, tokenizers and data.

```bash
bash ./download.sh
```

unzip and use my trained mT5 to predict result.

```bash
bash ./run.sh ./data/public.jsonl ./submission.jsonl
```

---

## Training

### Start Training

for example:

```bash
python run_summarization_no_trainer.py \
--model_name_or_path google/mt5-small \
--max_source_length 256 \
--max_target_length 64 \
--text_column maintext \
--summary_column title \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 8 \
--num_train_epochs 15 \
--learning_rate 3e-4 \
--num_beams 4 \
--train_file ./data/train.jsonl \
--validation_file ./data/public.jsonl \
--seed 114514 \
--output_dir ./tmp/1029
```

- `model_name_or_path`: Path to pretrained model or model from huggingface.co/models.
- `train_file`: Path to `train.jsonl`.
- `validation_file`: Path to `public.jsonl`.
- `output_dir`: The output directory where the model will be stored.

---

## Testing

```bash
python inference.py \
--model ./tmp/1029 \
--max_length 64 \
--beam_size 4 \
--top_k 20 \
--top_p 0.85 \
--test_data ./data/public.jsonl \
--output_dir ./tmp/1029
```

- `model_name_or_path`: Path to pretrained model.
- `test_data`: Path to testing data.
- `output_dir`: The output path where the result will be stored（jsonl format）.

## Final Report
The final report of this project provides a comprehensive overview of the project, including the following sections:

1. **Model**

2. **Training**

3. **Generation Strategies**

For more detailed information, please refer to the [full report](./report.pdf).