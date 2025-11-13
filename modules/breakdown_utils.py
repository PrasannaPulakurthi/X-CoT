import json

def load_video_breakdowns(path):
    with open(path, "r", encoding="utf-8") as f:
        video_breakdowns = [json.loads(line) for line in f]

    caps = {}
    for vb in video_breakdowns:
        vid = vb.get("video_id", "unknown_id")
        parts = []

        # 1) summary
        if summ := vb.get("summary"):         
            for i, s in enumerate(summ, start=1):
                parts.append(f"(1) Summary {i}: {s}")
            # parts.append(f"(2) Video Summary: {summ}")
        
        # 2) key objects (first 5 if list is long)
        objs = vb.get("objects")
        if isinstance(objs, list) and objs:
            # filtered = objs[:5]
            parts.append(f"(2) Objects: {', '.join(objs)}")

        # 3) key actions (first 5 if list is long)
        actions = vb.get("actions")
        if isinstance(actions, list) and actions:
            # filtered = actions[:5]
            parts.append(f"(3) Actions: {', '.join(actions)}")

        # 4) high-level scene
        if scenes := vb.get("scenes"):
            # parts.append(f"(5) Scene: {scene}")
            parts.append(f"(4) Scenes: {', '.join(scenes)}")

        # 5) frame-by-frame captions
        frames = vb.get("frame_captions", [])
        parts.append(f"(5) Frame Captions:")
        for i, caption in enumerate(frames, start=1):
            if isinstance(caption, str) and caption.strip():
                parts.append(f"{caption.strip()}")

        # Final composition
        caps[vid] = ["\n".join(parts)]
        # print(vid,caps[vid])
        
    return caps

