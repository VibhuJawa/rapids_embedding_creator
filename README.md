# RAPIDS EMBEDDING CREATOR

## Description
This is a tool that allows you to create embeddings on a given dataset using RAPIDS, Sentence Transformers and Pytorch. 


### Installation:

Conda
```bash
mamba create -n rapids_23_08_sentence_transformers -c rapidsai-nightly -c nvidia -c conda-forge pytorch-cuda=11.8  cudatoolkit=11.8 python=3.9 pytorch-nightly torchvision torchaudio sentence-transformers rapids=23.08
```

### Run Workflow

```bash
python3 embedding_creation.py --input_file_name bb-string-df.parquet --output_file_name "/raid/vjawa/bb-string-embeddings-df.parquet" --rmm_pool_size "12GB" --CUDA_VISIBLE_DEVICES='0,1,2,3'
```