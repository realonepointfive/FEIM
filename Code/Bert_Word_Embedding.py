import pandas as pd
import os
from transformers import BertTokenizer

curr_dir = os.path.dirname(__file__)
rela_dir = "../Dataset/Flood_input_data/"
dataset_path = os.path.normpath(os.path.join(curr_dir, rela_dir))

df = pd.read_csv(dataset_path + '/512to521MsgTextSample.txt', header=None, names=['sentence'])
sentences = df.sentence.values

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Print the original sentence.
print(' Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))


