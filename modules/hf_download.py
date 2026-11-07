import os

HF_REPO_ID = "prasannareddyp/X-CoT"
HF_DATASETS = ["MSRVTT", "MSVD", "DiDeMo", "LSMDC"]
HF_RETRIEVERS = ["clip", "vlm2vec", "xpool"]


def ensure_hf_file(filename, local_dir="./"):
    """Download `filename` (e.g. outputs/MSVD/xpool_ranking_benchmark.jsonl) from the
    X-CoT Hugging Face dataset into `local_dir` if it is not already present."""
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        return local_path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {filename} from Hugging Face ({HF_REPO_ID}).")
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename,
                           repo_type="dataset", local_dir=local_dir)


def ensure_xcot_inputs(dataset_name, retrieval_method, test_mode="benchmark", local_dir="./"):
    """Make sure the initial retriever ranking and video breakdowns exist locally,
    downloading them from Hugging Face when missing. Returns (ranks_path, breakdowns_path)."""
    ranks_file = f"outputs/{dataset_name}/{retrieval_method}_ranking_{test_mode}.jsonl"
    breakdowns_file = f"outputs/{dataset_name}/video_breakdowns_{test_mode}.jsonl"

    hosted = (dataset_name in HF_DATASETS and test_mode == "benchmark")
    for f in (breakdowns_file, ranks_file):
        if os.path.exists(os.path.join(local_dir, f)):
            continue
        if not hosted or (f == ranks_file and retrieval_method not in HF_RETRIEVERS):
            raise FileNotFoundError(
                f"{f} not found and not available on Hugging Face. "
                f"Hosted: datasets {HF_DATASETS}, retrievers {HF_RETRIEVERS}, test_mode 'benchmark'. "
                f"Generate it with xpool_ranking.py / vlm2vec_ranking.py (see initial_retrievers_cmds.sh).")
        ensure_hf_file(f, local_dir)

    return os.path.join(local_dir, ranks_file), os.path.join(local_dir, breakdowns_file)
