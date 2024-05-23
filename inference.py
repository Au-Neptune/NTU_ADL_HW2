import os, torch, argparse
from tqdm import tqdm
from tools.utils import convert_data_to_jsonl
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference arguments for mT5 model.")
    parser.add_argument("--model", type=str, help="The path of the model.")
    parser.add_argument("--test_data", type=str, help="The name of the data.")
    parser.add_argument("--max_length", type=int, default=64,
                        help="The output max length.")
    parser.add_argument("--beam_size", type=int,
                        default=4, help="The beam size.")
    parser.add_argument("--top_p", type=float,
                        default=0.85, help="The top p score.")
    parser.add_argument("--top_k", type=int, default=20,
                        help="The top k score.")
    parser.add_argument("--output_dir", type=str,
                        help="The name of the output directory.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def preprocess_function(examples):
        inputs = examples["maintext"]
        model_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True
        )
        return model_inputs

    data_files = {}
    data_files["test"] = args.test_data
    raw_datasets = load_dataset('json', data_files=data_files)

    ids = raw_datasets["test"]["id"]

    test_dataset = raw_datasets["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["test"].column_names
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=24
    )

    gen_kwargs = {
        "max_length": args.max_length,
        "num_beams": args.beam_size,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": True,
    }

    model.to("cuda")
    model.eval()
    result = []
    counter = 0

    for step, batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                **gen_kwargs
            )

            decoded_preds = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

            for i in range(len(decoded_preds)):
                result.append({"title": decoded_preds[i], "id": ids[counter]})
                counter += 1

    convert_data_to_jsonl(result, os.path.join(
        args.output_dir, "submission.jsonl"))


if __name__ == "__main__":
    main()
