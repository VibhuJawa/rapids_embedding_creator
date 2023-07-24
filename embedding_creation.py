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

import cupy as cp
import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer
from cudf.core.column import as_column
from sentence_transformers import SentenceTransformer
from dask.distributed import get_worker, performance_report
import dask_cudf
import time
from cluster_setup import setup_dask_cluster
from rapids_embeddings_helpers import create_embeddings as create_embeddings_rapids
import gc


# Embedding creation workflow
def create_list_series_from_2d_ar(ar, index):
    """
    Create a cudf list series  from 2d arrays
    """
    n_rows, n_cols = ar.shape
    data = as_column(ar.flatten())
    offset_col = as_column(
        cp.arange(start=0, stop=len(data) + 1, step=n_cols), dtype="int32"
    )
    mask_col = cp.full(shape=n_rows, fill_value=True)
    mask = cudf._lib.transform.bools_to_mask(as_column(mask_col))
    lc = cudf.core.column.ListColumn(
        size=n_rows,
        dtype=cudf.ListDtype(data.dtype),
        mask=mask,
        offset=0,
        null_count=0,
        children=(offset_col, data),
    )
    return cudf.Series(lc, index=index)


def add_embedding(df, batch_size, use_rapids_tokenizer=False):
    """
    This function runs the entire ner workflow end2end on a single GPU
    """

    worker = get_worker()
    if hasattr(worker, "sbert_model"):
        model = worker.sbert_model
    else:
        print(f"Loading Model on {worker.id}", flush=True)
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        worker.sbert_model = model.to("cuda")

    if use_rapids_tokenizer:
        if hasattr(worker, 'rapids_tokenizer'):
            cudf_tokenizer = worker.rapids_tokenizer
        else:
            # Vocabulary is included in the root directory of this repo
            # however, below is the command to modify / update it -->
            # from cudf.utils.hash_vocab_utils import hash_vocab
            # hash_vocab('vocab.txt', 'voc_hash.txt')
            cudf_tokenizer = SubwordTokenizer("/home/nfs/vjawa/rapids_embedding_creator/vocab/voc_hash.txt", do_lower_case=True)
            worker.rapids_tokenizer = cudf_tokenizer

        embedding = create_embeddings_rapids(df["String"], model, cudf_tokenizer, batch_size)
    else:
        embedding = model.encode(
            df["String"].to_arrow().to_pylist(),
            batch_size=batch_size,
            show_progress_bar=True,
            device="cuda",
            convert_to_tensor=True,
        )
    embedding = cp.asarray(embedding)
    df["embeddings"] = create_list_series_from_2d_ar(embedding, df.index)
    gc.collect()

    return df


def embedding_creation_workflow(
    input_file_name, output_file_name, batch_size, use_rapids_tokenizer
):
    """
    This function runs the entire embedding creation workflow end2end on multiple GPUs
        Args:
            input_file_name: The name of the input file.
            output_file_name: The name of the output file.
            batch_size: The batch size to be used for the embedding creation.
    """
    df = dask_cudf.read_parquet(input_file_name)
    #TODO Make `repartition` configurable
    df = df.repartition(128)
    meta_df = df._meta.copy()
    meta_df["embeddings"] = [1] * len(meta_df)
    df = df.map_partitions(
        add_embedding,
        batch_size=batch_size,
        use_rapids_tokenizer=use_rapids_tokenizer,
        meta=meta_df,
    )
    df.to_parquet(output_file_name, write_index=False)


def parse_args():
    """
    This function parses the command line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the embedding creation workflow end2end on multiple GPUs"
    )
    parser.add_argument(
        "--input_file_name", type=str, help="The name of the input file.", required=True
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="The name of the output file.",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size to be used for the embedding creation.",
        default=1024 * 2,
    )
    parser.add_argument(
        "--use_rapids_tokenizer",
        action="store_true",
        help="Whether to use rapids tokenizer or not.",
    )
    parser.add_argument(
        "--CUDA_VISIBLE_DEVICES", type=str, help="The GPUs to be used by the cluster."
    )
    parser.add_argument(
        "--rmm_pool_size",
        type=str,
        help="The size of the RMM pool to be used by each worker.",
    )
    parser.add_argument(
        "--create-dask-profile",
        action="store_true",
        help="Whether to create a dask profile or not.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    client = setup_dask_cluster(args.rmm_pool_size, args.CUDA_VISIBLE_DEVICES)
    st = time.time()
    if args.create_dask_profile:
        with performance_report(filename="dask-embedding-creation.html"):
            embedding_creation_workflow(
                args.input_file_name,
                args.output_file_name,
                args.batch_size,
                args.use_rapids_tokenizer,
            )
    else:
        embedding_creation_workflow(
            args.input_file_name,
            args.output_file_name,
            args.batch_size,
            args.use_rapids_tokenizer,
        )
    et = time.time()
    print(f"Total time taken: {et-st}")
