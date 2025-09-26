import pandas as pd
def segments_to_submission_rows(video_id, agent_id, target_id, segs):
    rows = []
    rid = 0
    for action, s, e in segs:
        rows.append({
            "row_id": rid, "video_id": video_id,
            "agent_id": agent_id, "target_id": target_id,
            "action": action, "start_frame": s, "stop_frame": e
        }); rid += 1
    return pd.DataFrame(rows)
