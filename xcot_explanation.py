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


def make_compare_prompt(query, descA, descB):
    return [
        {"role": "system", "content": (
            "You are a retrieval expert. Given a query, compare two video breakdowns and decide which is a better match. "
            "You are provided with Frame Captions, three Video Summaries, and additional video breakdowns such as Objects, Actions, Scenes. "
            "Evaluate both videos only based on semantic alignment with the query.\n"
            "Do NOT favor writing style or position.\n"
            "Respond strictly in the following two-line format:\n"
            "Answer: A or B\n"
            "Reason: One or two sentences explaining why the chosen video matches better."
        )},
        {"role": "user", "content": f"Query: {query}"},
        {"role": "user", "content": f"Video A: {descA}"},
        {"role": "user", "content": f"Video B: {descB}"},
        {"role": "user", "content": (
            "Which video better matches the query? Respond strictly in the format:\n"
            "Answer: A or B\n"
            "Reason: One or two sentences explanation."
        )}
    ]

def LLM_Reason(response_pairs, order_bt, chat, k):
    message = [
        {"role": "user", "content": (
            f"Final order of videos: {order_bt}\n"
            f"Total Number of videos: {k}\n"
            f"Pairwise Reasons: {response_pairs}\n"
            "Given the above information Answer in the following format: "
            f"Summary: Please provide a high-level summary one or two sentences explaining why the Video {order_bt[0]} was selected as Top-1 over the rest.\n"
        )}]
    response = chat(message, max_new_tokens=200).strip()
    return response


def evaluate_llm_reranking(cfg):
    log_name = f'xcot_reason_{cfg.dataset_name}_{cfg.topk_retrieval_method}'
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
    response_pairs = defaultdict(list)
    seen_pairs     = defaultdict(set)
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

    for i, query in tqdm(enumerate(queries), total=len(queries)):
        full_vid_idx = rank_list_idx[i]
        vid_idx = full_vid_idx[:cfg.topk]
        # vid_ids = [queries_id[j] for j in vid_idx]
        vid_ids = [index_to_video_id[j] for j in vid_idx]
        # true = queries_id[i]

        start_time = time.perf_counter()  # <-- start timer

        def extract_answer(text):
            match = re.search(r'^\s*Answer:\s*([A-Za-z0-9]+)\b', text, re.IGNORECASE | re.MULTILINE)
            return match.group(1) if match else None

        @lru_cache(maxsize=10000)
        def _compare(a: str, b: str, query: str) -> str:
            prompt = make_compare_prompt(query, breakdowns[a], breakdowns[b])
            response = chat(prompt,max_new_tokens=100).strip()
            return (response)
    
        def cmp(a_vid, b_vid, *, a_idx=None, b_idx=None):
            response = _compare(a_vid, b_vid, query)
            letter   = extract_answer(response)
            if letter not in ("A", "B"):
                letter = "A"       # fallback

            # -----------------------  uniq-pair filter  ----------------------------
            if a_idx is not None and b_idx is not None:
                pair_key = tuple(sorted((a_idx, b_idx)))   # unordered identity
                if pair_key not in seen_pairs[i]:          # i = current query slice
                    seen_pairs[i].add(pair_key)            # remember it

                    # ------------- store orientation-aware result -----------------
                    if letter == "A":                      # A beats B
                        response_pairs[i].append((f"Video {a_idx+1} > Video {b_idx+1}", response)) # +1 (0 to N-1 -> 1 to N)
                        pair_logs[i].append((a_idx, b_idx))
                    else:                                  # B beats A
                        response_pairs[i].append((f"Video {a_idx+1} < Video {b_idx+1}", response)) # +1 (0 to N-1 -> 1 to N)
                        pair_logs[i].append((b_idx, a_idx))
            # ----------------------------------------------------------------------

            return letter

        if queries_idx[i] not in vid_idx:
            # The results is not present in the top_k, then skip re-ordering 
            ordered = vid_ids.copy()
            LLM_indices = [video_id_to_index[v] for v in ordered]
        else:
            # get re-ordered vids via sliding window 
            ordered = sliding_rank(vid_ids, cmp, passes=cfg.passes) 
            
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
                    order_bt = ordered
                # --------------------------------------------------------------------
            else:
                LLM_indices = [video_id_to_index[v] for v in ordered]

            order_bt_new = [x + 1 for x in order_bt] # +1 (0 to N-1 -> 1 to N)
            reason_summary = LLM_Reason(response_pairs[i],order_bt_new, chat, cfg.topk)
            pairwise_explanations = "\n".join(f"**({p[0]})** | {p[1]}" if isinstance(p, tuple) and len(p) == 2 else str(p) for p in response_pairs[i])

            # Print the Pairwise Response and the final summary
            gen_log(model_path=cfg.tb_log_dir, log_name=log_name, 
                    msg=f"[{i}] TopK: {vid_idx} \n\n**| Pairwise Explanations |** \n{pairwise_explanations} \n\n**| X-CoT Explanation Summary |**\n{reason_summary} \n\n ")

        LLM_indices.extend(full_vid_idx[cfg.topk:])
        LLM_pred_idx.append(LLM_indices)
        
        # Print the reordered video IDs
        gen_log(model_path=cfg.tb_log_dir, log_name=log_name, 
                msg=f"[{i}] TopK: {vid_idx} | GT: {queries_idx[i]} | LLM Pred Order: {LLM_indices[:cfg.topk]}")
        

        # compute recalls by position
        res = compute_reranked_recalls([queries_idx[i]], [LLM_indices])
        
        r1+=res['R1']; r5+=res['R5']; r10+=res['R10']
        end_time = time.perf_counter()  # <-- end timer
        elapsed = end_time - start_time
        total_elapsed += elapsed

        
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

if __name__ == "__main__":
    cfg = AllConfig()
    evaluate_llm_reranking(cfg)
