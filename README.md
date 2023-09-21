# RAPIDS EMBEDDING CREATOR

## Description
This is a tool that allows you to create embeddings on a given dataset using RAPIDS, Sentence Transformers and Pytorch. 


### Installation:

Conda
```bash
mamba create -n rapids_23_10_sentence_transformers \
    -c rapidsai-nightly \
    -c pytorch\
    -c nvidia \
    -c conda-forge \
    pytorch-cuda=11.8 \
    cudatoolkit=11.8 \
    python=3.9 \
    pytorch \
    torchvision \
    torchaudio \
    sentence-transformers \
    rapids=23.10
```

### Run Workflow

```bash
python3 embedding_creation.py \
    --input_file_name string-df.parquet \
    --output_file_name "/raid/vjawa/bb-string-embeddings-df.parquet" \
    --rmm_pool_size "12GB" \
    --CUDA_VISIBLE_DEVICES='0,1,2,3'
```
