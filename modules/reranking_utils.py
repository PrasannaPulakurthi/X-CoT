import torch
import json
import os
from modules.metrics import compute_metrics
import numpy as np

def load_ranking_info(json_path):
    results = []
    with open(json_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            results.append(record)
    return results

def save_ranking_info(config, sims, all_vid_ids, all_texts, output_path):
    unique_vid_ids = list(dict.fromkeys(all_vid_ids))         # preserves order
    vid_id_to_col  = {v: i for i, v in enumerate(unique_vid_ids)}

    all_scores, all_indices = torch.sort(sims, dim=1, descending=True)

    # Create output directory if not exists
    os.makedirs(f"outputs/{config.dataset_name}", exist_ok=True)

    # Save one JSON object per line
    with open(output_path, 'w') as f:
        for query_idx, (indices, scores) in enumerate(zip(all_indices, all_scores)):
            gt_vid_id  = all_vid_ids[query_idx] 
            record = {
                "video_index": vid_id_to_col[gt_vid_id],
                "caption_index": query_idx,
                "video_id": gt_vid_id,
                "GT Caption": all_texts[query_idx],
                "ranking_indices": indices.tolist(),
                "ranking_scores": scores.tolist()
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved full retrieval results in JSONL format to {output_path}")
    print(f"Num Text: {sims.shape[0]}, Num Videos: {sims.shape[1]}")


def compute_reranked_recalls(query_ids, ranking_indices):
    ranks = []
    for i, gt_id in enumerate(query_ids):
        ranked_list = ranking_indices[i]
        if gt_id in ranked_list:
            ranks.append(ranked_list.index(gt_id))
    return compute_metrics(np.array(ranks))