from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from util.kmp import kmp
from megatron.data.indexed_dataset import MMapIndexedDatasetBuilder



"""
path = '/mnt/nvme/home/honglu/diff_data/tokenized_filtered_deduped_diff_data'
dataset = load_dataset("parquet", data_files={'train': f'{path}/train.parquet', 'test': f'{path}/validation.parquet'})

dff_tokens = [27, 35, 5777, 29] 
def gen_labels(examples):
    labels = []
    for i in range(len(examples['input_ids'])):
        assert len(examples['input_ids'][i]) == len(examples['attention_mask'][i])
        # Start searching for <DFF>
        tokens = examples['input_ids'][i]
        index = kmp(tokens, dff_tokens, first_appearance=True)
        if index:
            index = index[0]
            labels.append([-100] * index + tokens[index:].copy())
        else:
            print(f"ERR: The {i}-th sample does not have <DFF>.")
            labels.append([-100])

    examples['labels'] = labels
    return examples
masked_dataset = dataset.map(
    gen_labels,
    batched=True,
    num_proc=50,
    load_from_cache_file=False,
    desc="Generating masked labels.",
)

print(dataset)
print(masked_dataset)
masked_dataset.save_to_disk('masked')
"""




vocab_size = 50295
eod = 50256
masked_dataset = load_from_disk('final_v2')

path = 'neox_diff_data'
split = 'validation'
keys = ['input_ids', 'labels']
out_keys = ['text', 'label']
dtypes = [np.int32, np.int32]

prefix = f"{path}/{split}"

output_bin_files = {}
output_idx_files = {}
builders = {}
for key, dtype, out_key in zip(keys, dtypes, out_keys):
    output_bin_files[key] = "{}_{}_{}.bin".format(
        prefix, out_key, "document"
    )
    output_idx_files[key] = "{}_{}_{}.idx".format(
        prefix, out_key, "document"
    )
    builders[key] = MMapIndexedDatasetBuilder(
        output_bin_files[key], 
        dtype=dtype
    )

for i in tqdm(range(len(masked_dataset[split]))):
    content = masked_dataset[split][i]
    #if content['attention_mask'][0] != 0:
    #    continue  # cut off the first token to leave place for EOS
    start = np.argmax(content['attention_mask'])

    for key, dtype in zip(keys, dtypes):
        #builders[key].add_item(np.concatenate([np.array(content[key][start:], dtype=dtype), np.array([eod], dtype=dtype)]))
        builders[key].add_item(np.array(content[key][start:] + [eod], dtype=dtype))
        builders[key].end_document()


for key in keys:
    builders[key].finalize(output_idx_files[key])

