from config.all_config import AllConfig
import re
import torch
import time
from functools import lru_cache
from config.all_config import gen_log
import numpy as np
import choix
from modules.breakdown_utils import load_video_breakdowns
from modules.reranking_utils import load_ranking_info, compute_reranked_recalls
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from collections import defaultdict, deque
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class LLMChat:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

        # Build a clean, greedy generation config (no sampling params set)
        self.gen_cfg = GenerationConfig(
            do_sample=False,                # greedy
            temperature=None,               # unset sampling knobs
            top_p=None,
            top_k=None,
            num_beams=1,
            max_new_tokens=10,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def __call__(self, messages, max_new_tokens=10):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        gen_cfg = self.gen_cfg
        gen_cfg.max_new_tokens = max_new_tokens

        out_ids = self.model.generate(**inputs, generation_config=gen_cfg)
        gen = out_ids[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def sliding_rank(vids, compare_fn, passes=3):
    vids = vids.copy()
    idxs = list(range(len(vids))) 
    n = len(vids)
    for _ in range(passes):
        for pos in range(0, n-1):
            if compare_fn(vids[pos], vids[pos+1], a_idx=idxs[pos], b_idx=idxs[pos + 1]) == "B":
                vids[pos], vids[pos+1] = vids[pos+1], vids[pos]
                idxs[pos], idxs[pos + 1] = idxs[pos + 1], idxs[pos]
    return vids


def sliding_rank_odd_even(vids, compare_fn, passes=3):
    """
    Odd-even version of sliding-rank.
    """
    vids = vids.copy()
    idxs = list(range(len(vids)))
    n = len(vids)
    for _ in range(passes):
        # ---------- EVEN phase: (0,1), (2,3), ... -----------------
        for i in range(0, n - 1, 2):
            if compare_fn(vids[i], vids[i + 1],
                          a_idx=idxs[i], b_idx=idxs[i + 1]) == "B":
                vids[i], vids[i + 1] = vids[i + 1], vids[i]
                idxs[i], idxs[i + 1] = idxs[i + 1], idxs[i]
        # ---------- ODD phase: (1,2), (3,4), ... -------------------
        for i in range(1, n - 1, 2):
            if compare_fn(vids[i], vids[i + 1],
                          a_idx=idxs[i], b_idx=idxs[i + 1]) == "B":
                vids[i], vids[i + 1] = vids[i + 1], vids[i]
                idxs[i], idxs[i + 1] = idxs[i + 1], idxs[i]
    return vids


def _phase_parallel(vids, idxs, start, compare_fn, max_workers=8):
    """
    One odd/even phase starting at `start` (0 for even, 1 for odd).
    Runs pairwise comparisons in parallel, then applies swaps in order.
    """
    n = len(vids)
    pairs = [(i, i+1) for i in range(start, n - 1, 2)]
    if not pairs:
        return vids, idxs

    # Dispatch comparisons in parallel (read-only access to vids/idxs)
    results = {}  # i -> "A"/"B"
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2i = {
            ex.submit(
                compare_fn,
                vids[i], vids[j],
                a_idx=idxs[i], b_idx=idxs[j]
            ): i
            for (i, j) in pairs
        }
        for fut in as_completed(fut2i):
            i = fut2i[fut]
            results[i] = fut.result()

    # Apply swaps sequentially to avoid race conditions
    for i, j in pairs:
        if results.get(i) == "B":
            vids[i], vids[j] = vids[j], vids[i]
            idxs[i], idxs[j] = idxs[j], idxs[i]

    return vids, idxs


def sliding_rank_odd_even_parallel(vids, compare_fn, passes=3, max_workers=8):
    """
    Odd-even sliding-rank with parallel comparisons per phase.
    """
    vids = vids.copy()
    idxs = list(range(len(vids)))
    for _ in range(passes):
        # EVEN phase: (0,1), (2,3), ...
        vids, idxs = _phase_parallel(vids, idxs, start=0, compare_fn=compare_fn, max_workers=max_workers)
        # ODD phase: (1,2), (3,4), ...
        vids, idxs = _phase_parallel(vids, idxs, start=1, compare_fn=compare_fn, max_workers=max_workers)
    return vids


def make_compare_prompt(query, descA, descB):
    return [
        {"role": "system", "content": (
            "You are a retrieval expert. Given a query, compare two video breakdowns and decide which is a better match. "
            "You are provided with Frame Captions, three Video Summaries, and additional video breakdowns such as Objects, Actions, Scenes."
            "Evaluate both videos only based on semantic alignment with the query.\n"
            "Do NOT favor writing style or position.\n"
            "Only respond with 'A' or 'B' (without explanation)."
        )},
        {"role": "user", "content": f"Query: {query}"},
        {"role": "user", "content": f"Video  A: {descA}"},
        {"role": "user", "content": f"Video  B: {descB}"},
        {"role": "user", "content": (
            "Which video better matches the query? Respond with 'A' or 'B' only."
        )}
    ]


def evaluate_llm_reranking(cfg):
    log_name = f'xcot_{cfg.dataset_name}_{cfg.topk_retrieval_method}'
    # @PRP: for logging on RC/local
    msg = f'model pth = {cfg.tb_log_dir}'
    gen_log(model_path=cfg.tb_log_dir, log_name=log_name, msg=msg)
    msg = f'\nconfig={cfg.__dict__}\n'
    gen_log(model_path=cfg.tb_log_dir, log_name=log_name, msg=msg)
    gen_log(model_path=cfg.tb_log_dir, log_name=log_name, msg='\nrecord all training and testing results\n')

    chat = LLMChat("Qwen/Qwen2.5-7B-Instruct-1M")
    # chat = LLMChatBig("Qwen/Qwen2.5-14B-Instruct-1M")

    # load data
    ranks_dict = load_ranking_info(
        f"outputs/{cfg.dataset_name}/{cfg.topk_retrieval_method}_ranking_{cfg.test_mode}.jsonl"
    )
    breakdowns = load_video_breakdowns(
        f"outputs/{cfg.dataset_name}/video_breakdowns_{cfg.test_mode}.jsonl"
    )

    # evaluation
    r1=r5=r10=0
    total_elapsed = 0
    LLM_pred_idx = []
    pair_logs = defaultdict(list)
    video_id_to_index = {}
    index_to_video_id = {}

    queries_idx          = [entry["video_index"] for entry in ranks_dict]       # 1
    queries_id           = [entry["video_id"] for entry in ranks_dict]          # "video7025"
    queries              = [entry["GT Caption"] for entry in ranks_dict]        # "a person is connecting something to system"
    rank_list_idx        = [entry["ranking_indices"] for entry in ranks_dict]   # [0, 13, 83, 91, 36, ....]

    for entry in ranks_dict:
        if entry["video_id"] not in video_id_to_index:
            video_id_to_index[entry["video_id"]] = entry["video_index"]
    for entry in ranks_dict:
        if entry["video_index"] not in index_to_video_id:
            index_to_video_id[entry["video_index"]] = entry["video_id"]

    overall_hits   = 0
    overall_misses = 0
    for i, query in tqdm(enumerate(queries), total=len(queries)):
        full_vid_idx = rank_list_idx[i]
        vid_idx = full_vid_idx[:cfg.topk]
        # vid_ids = [queries_id[j] for j in vid_idx]
        vid_ids = [index_to_video_id[j] for j in vid_idx]
        # true = queries_id[i]

        start_time = time.perf_counter()  # <-- start timer

        @lru_cache(maxsize=10000)
        def _compare(a: str, b: str, query: str) -> str:
            prompt = make_compare_prompt(query, breakdowns[a], breakdowns[b])
            response = chat(prompt,max_new_tokens=4).strip()
            return (response)
    
        # Clear any stale statistics
        _compare.cache_clear()           #  <-- optional but keeps counts clean
        start_cache = _compare.cache_info()   # hits=0, misses=0
    
        def cmp(a_vid, b_vid, *, a_idx=None, b_idx=None):
            letter = _compare(a_vid, b_vid, query)
            # letter = parse_ranked_order(response)
            # print(response, letter)
            if letter not in ("A", "B"):
                letter = "A"       # fallback

            # log result for BT -------------------------------------------------
            if a_idx is not None and b_idx is not None:
                if letter == "A":
                    pair_logs[i].append((a_idx, b_idx))   # A beat B
                else:
                    pair_logs[i].append((b_idx, a_idx))   # B beat A
            # ------------------------------------------------------------------

            return letter

        if queries_idx[i] not in vid_idx:
            # The results is not present in the top_k, then skip re-ordering 
            ordered = vid_ids.copy()
            LLM_indices = [video_id_to_index[v] for v in ordered]
        else:
            if cfg.sort_type == "sliding_window_parallel":
                # get re-ordered vids via sliding window 
                ordered = sliding_rank_odd_even_parallel(vid_ids, cmp, passes=cfg.passes) 
            else:
                ordered = sliding_rank_odd_even(vid_ids, cmp, passes=cfg.passes) 

            # --------------------------------------------------------------------
            K = len(vid_ids)
            if pair_logs[i]:                         # at least one comparison happened
                try:
                    # ILSR solver – small gamma prior for stability
                    bt_scores = choix.ilsr_pairwise(n_items=K,
                                                    data=pair_logs[i],
                                                    alpha=1e-3)
                    order_bt = np.argsort(-bt_scores)        # indices 0..K-1
                    # ordered = [vid_ids[j] for j in order_bt]    # replace sliding-rank output
                    LLM_indices = [vid_idx[v] for v in order_bt]
                except RuntimeError as e:
                    LLM_indices = [video_id_to_index[v] for v in ordered]
                # --------------------------------------------------------------------
            else:
                LLM_indices = [video_id_to_index[v] for v in ordered]

        LLM_indices.extend(full_vid_idx[cfg.topk:])
        LLM_pred_idx.append(LLM_indices)
        
        # Print the reordered video IDs
        gen_log(model_path=cfg.tb_log_dir, log_name=log_name, 
                msg=f"[{i}] TopK: {vid_idx} | GT: {queries_idx[i]} | LLM Pred Order: {LLM_indices[:cfg.topk]}")
        
        stats = _compare.cache_info()     # namedtuple(hits, misses, maxsize, currsize)
        new_hits   = stats.hits   - start_cache.hits
        new_misses = stats.misses - start_cache.misses
        total_lookups = new_hits + new_misses
        overall_hits   += new_hits
        overall_misses += new_misses
        hit_rate = 100.0 * new_hits / total_lookups if total_lookups else 0.0
        print(f"LRU-cache stats -> hits: {new_hits:,}  "
            f"misses: {new_misses:,}  "
            f"hit-rate: {hit_rate:5.1f}%")
        
        end_time = time.perf_counter()  # <-- end timer
        elapsed = end_time - start_time
        total_elapsed += elapsed

        # compute recalls by position
        res = compute_reranked_recalls([queries_idx[i]], [LLM_indices])
        
        r1+=res['R1']; r5+=res['R5']; r10+=res['R10']

        
        gen_log(model_path=cfg.tb_log_dir, log_name=log_name, 
                msg=(
                    f"R@1={r1/(i+1):5.2f}% "
                    f"R@5={r5/(i+1):5.2f}% "
                    f"R@10={r10/(i+1):5.2f}% "
                    f"Time taken: {elapsed:.2f} seconds"
                    ))

    res = compute_reranked_recalls(queries_idx, LLM_pred_idx)
    window_metric = defaultdict(lambda: deque(maxlen=cfg.eval_window_size))
    # Compute window metrics
    for m in res:
        window_metric[m].append(res[m])

    # Compute average of window metrics
    for m in window_metric:
        res[m + "-window"] = np.mean(window_metric[m])

    gen_log(model_path=cfg.tb_log_dir, log_name=log_name, 
            msg=(f"-------------------------------------\n",
        f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
        f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
        f"R@10: {res['R10']} (window: {res['R10-window']})\n",
        f"R@20: {res['R20']} (window: {res['R20-window']})\n",
        f"R@50: {res['R50']} (window: {res['R50-window']})\n",
        f"R@100: {res['R100']} (window: {res['R100-window']})\n",             
        f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
        f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
        f"--------Time Taken: {total_elapsed:.2f} seconds-----\n"))

    total = overall_hits + overall_misses
    if total:
        overall_rate = 100.0 * overall_hits / total
        print(f"\n==========  GLOBAL LRU-CACHE  =================")
        print(f"hits   : {overall_hits:,}")
        print(f"misses : {overall_misses:,}")
        print(f"hit-rate: {overall_rate:5.1f}%")
        gen_log(model_path=cfg.tb_log_dir,
                log_name=log_name,
                msg=(f"GLOBAL cache — hits {overall_hits:,}, "
                    f"misses {overall_misses:,}, "
                    f"hit-rate {overall_rate:5.1f}%"))
    else:
        print("No look-ups recorded – something is wrong with the instrumentation.")
        
if __name__ == "__main__":
    cfg = AllConfig()
    evaluate_llm_reranking(cfg)
