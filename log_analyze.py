# log_analyze.py
import re
import sys
import pandas as pd

EP_PATTERN = re.compile(
    r"\[EP (?P<ep>\d+)\]\s+len=(?P<len>\d+)\s+R=(?P<R>[-+]?\d+\.?\d*)\s+danger=(?P<danger>\d+)\s+blocked=(?P<blocked>\d+)"
)
CKPT_PATTERN = re.compile(
    r"\[CKPT\]\s+ep=(?P<ep>\d+)\s+avg_len=(?P<avg_len>[-+]?\d+\.?\d*)\s+max_len=(?P<max_len>\d+)\s+epsilon=(?P<epsilon>[-+]?\d+\.?\d*)"
)

def main():
    if len(sys.argv) < 2:
        print("usage: python log_analyze.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]
    ep_rows = []
    ckpt_rows = []

    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            m = EP_PATTERN.search(line)
            if m:
                ep_rows.append({
                    "ep": int(m.group("ep")),
                    "len": int(m.group("len")),
                    "R": float(m.group("R")),
                    "danger": int(m.group("danger")),
                    "blocked": int(m.group("blocked")),
                })
                continue
            m = CKPT_PATTERN.search(line)
            if m:
                ckpt_rows.append({
                    "ep": int(m.group("ep")),
                    "avg_len": float(m.group("avg_len")),
                    "max_len": int(m.group("max_len")),
                    "epsilon": float(m.group("epsilon")),
                })

    ep_df = pd.DataFrame(ep_rows)
    ckpt_df = pd.DataFrame(ckpt_rows)

    print("=== EPISODE STATS ===")
    if not ep_df.empty:
        print(ep_df.describe())
        # 최근 200에피 평균 길이
        print("\n최근 200에피 평균 길이:")
        print(ep_df.tail(200)["len"].mean())
    else:
        print("no episode lines")

    print("\n=== CKPT STATS ===")
    if not ckpt_df.empty:
        print(ckpt_df)
    else:
        print("no ckpt lines")

if __name__ == "__main__":
    main()
