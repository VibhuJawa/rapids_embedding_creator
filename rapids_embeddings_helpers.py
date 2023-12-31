# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cudf.core.subword_tokenizer import _cast_to_appropriate_type
import torch




def tokenize_strings(sentences, tokenizer, max_length):
    # Tokenize cudf Series
    token_o = tokenizer(
        sentences,
        max_length=max_length,
        max_num_rows=len(sentences),
        padding="max_length",
        return_tensors="cp",
        truncation=True,
        add_special_tokens=True,
    )
    return token_o 


def clip_tokens(token_o, max_length):
    clip_len = max_length - int((token_o["input_ids"][:, ::-1] != 0).argmax(1).min())
    token_o["input_ids"] = _cast_to_appropriate_type(
        token_o["input_ids"][:, :clip_len], "pt"
    )
    token_o["attention_mask"] = _cast_to_appropriate_type(
        token_o["attention_mask"][:, :clip_len], "pt"
    )

    del token_o["metadata"]
    return token_o


def create_embeddings(sentences, embedding_model, tokenizer, batch_size):
    """Creates the sentence embeddings using SentenceTransformer

    Args:
        sentences (cudf.Series[str]): a cuDF Series of Input strings

    Returns:
        embeddings (cupy.ndarray): corresponding sentence
        embeddings for the strings passed
    """
    all_embeddings_ls = []
    max_length = 1024
    tokenized_d = tokenize_strings(sentences, tokenizer, max_length).copy()
    with torch.no_grad():
        for s_ind in range(0, len(sentences), batch_size):
            e_ind = min(s_ind + batch_size, len(sentences))
            b_tokens = {k:v[s_ind:e_ind] for k,v in tokenized_d.items()}
            b_tokens = clip_tokens(b_tokens, max_length)
            model_output = embedding_model.forward(b_tokens)
            all_embeddings_ls.append(model_output["sentence_embedding"])

    all_embeddings = torch.vstack(all_embeddings_ls)
    return all_embeddings
