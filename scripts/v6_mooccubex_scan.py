"""Version 4.3.0. Resumable full pass over MOOCCubeX relations/user-problem.json by HTTP range requests.
Keeps records on problems linked to a concept of the CS discipline; counts all records.
Run in a working directory that holds relations/concept-problem.txt as concept-problem.txt; writes
cs_records2.jsonl, state.json (resume point) and scan_summary.json there.
Usage: python scripts/v6_mooccubex_scan.py   (then scripts/v6_mooccubex_log.py)"""
import sys, json, time, os, subprocess
URL = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2/relations/user-problem.json"
SIZE = 22453311654; CHUNK = 256 * 1024 * 1024
cs = set()
for line in open("concept-problem.txt", encoding="utf-8"):
    cid, pid = line.rstrip("\n").split("\t")[:2]
    if cid.startswith("K_") and cid.rsplit("_", 1)[1] == "计算机科学与技术":
        cs.add(pid)
st = json.load(open("state.json")) if os.path.exists("state.json") else {"offset": 0, "records": 0, "kept": 0, "users": 0}
out = open("cs_records2.jsonl", "a", encoding="utf-8")
key = b'"problem_id": "'
csb = {p.encode() for p in cs}
carry = b""
t0 = time.time()
while st["offset"] < SIZE:
    end = min(SIZE, st["offset"] + CHUNK) - 1
    for attempt in range(8):
        r = subprocess.run(["curl", "-sS", "-f", "-r", f"{st['offset']}-{end}", URL], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == end - st["offset"] + 1:
            break
        time.sleep(5 * (attempt + 1))
    else:
        sys.exit("failed at %d" % st["offset"])
    buf = carry + r.stdout
    lines = buf.split(b"\n")
    carry = lines.pop()  # incomplete last line
    for s in lines:
        if not s:
            continue
        st["records"] += 1
        i = s.find(key)
        if i >= 0:
            j = s.find(b'"', i + len(key))
            if s[i + len(key):j] in csb:
                out.write(s.decode("utf-8", "replace") + "\n"); st["kept"] += 1
    out.flush()
    st["offset"] = end + 1
    st["carry"] = len(carry)
    json.dump(st, open("state.json", "w"))
    print(st["offset"], st["records"], st["kept"], round(time.time() - t0), flush=True)
if carry.strip():
    st["records"] += 1
json.dump({"records": st["records"], "kept": st["kept"], "cs_problems": len(cs), "bytes": SIZE}, open("scan_summary.json", "w"))
print("done", st)
