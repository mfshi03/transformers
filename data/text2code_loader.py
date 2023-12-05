from datasets import load_dataset

dataset = load_dataset("code_x_glue_tc_text_to_code", split="train")
count = 0
for data in dataset:
    if count > 10000:
        break
    with open("code.txt", "a") as f:
        s = data["nl"] + "[SEP]" + data["code"] + "\n"
        f.write(s)

    count += 1
