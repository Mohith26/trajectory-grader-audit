"""Extract the public LMSYS MT-Bench human-judgment data into compact JSON.

Inputs (public, CC BY 4.0, downloaded from Hugging Face):
  human-*.parquet        lmsys/mt_bench_human_judgments (split "human", 3,355 expert votes)
  gpt4_pair-*.parquet    lmsys/mt_bench_human_judgments (split "gpt4_pair", GPT-4 pairwise verdicts)
  gpt4_single.jsonl      lmsys/mt-bench space, data/mt_bench/model_judgment/gpt-4_single.jsonl
  question.jsonl         lmsys/mt-bench space, data/mt_bench/question.jsonl
Requires pyarrow only for reading the parquet file. Output is committed, so the app
and the tests never need this script.
"""
import collections, json, sys
import pyarrow.parquet as pq

src = sys.argv[1] if len(sys.argv) > 1 else "."
out = sys.argv[2] if len(sys.argv) > 2 else "data"
human = pq.read_table(f"{src}/human-00000-of-00001-25f4910818759289.parquet").to_pylist()
single = [json.loads(l) for l in open(f"{src}/gpt4_single.jsonl")]
pair = {}
for r in pq.read_table(f"{src}/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet").to_pylist():
    w = {"model_a": "a", "model_b": "b", "tie": "tie", "tie (inconsistent)": "inc"}[r["winner"]]
    pair[(int(r["question_id"]), int(r["turn"]), r["model_a"], r["model_b"])] = w
    pair[(int(r["question_id"]), int(r["turn"]), r["model_b"], r["model_a"])] = {"a": "b", "b": "a"}.get(w, w)
questions = {q["question_id"]: q for q in (json.loads(l) for l in open(f"{src}/question.jsonl"))}

score, judgment = {}, {}
for s in single:
    k = (int(s["question_id"]), s["model"], int(s["turn"]))
    score[k] = s["score"]
    judgment[k] = s["judgment"]

votes = collections.defaultdict(lambda: {"a": 0, "b": 0, "tie": 0})
answers = {}
for r in human:
    qid, turn, a, b = int(r["question_id"]), int(r["turn"]), r["model_a"], r["model_b"]
    w = {"model_a": "a", "model_b": "b"}.get(r["winner"], "tie")
    votes[(qid, turn, a, b)][w] += 1
    for m, conv in ((a, r["conversation_a"]), (b, r["conversation_b"])):
        msgs = [x["content"] for x in conv if x["role"] == "assistant"]
        if len(msgs) >= 1:
            answers.setdefault(f"{qid}|{m}", msgs)

items, used = [], set()
for (qid, turn, a, b), v in sorted(votes.items()):
    ka, kb = (qid, a, turn), (qid, b, turn)
    if ka not in score or kb not in score or score[ka] < 1 or score[kb] < 1:
        continue  # vicuna-13b-v1.2 has no public gpt-4 single score; skip those pairs
    expert = "a" if v["a"] > v["b"] else "b" if v["b"] > v["a"] else "tie"
    items.append({"qid": qid, "cat": questions[qid]["category"], "turn": turn, "a": a, "b": b,
                  "sa": score[ka], "sb": score[kb], "pw": pair[(qid, turn, a, b)], "votes": [v["a"], v["b"], v["tie"]], "expert": expert})
    used.update([ka, kb])

tmpl = single[0]["user_prompt"]
tmpl = tmpl[: tmpl.index("[Question]")].strip()
texts = {
    "prompt_single": tmpl,
    "questions": {str(q): questions[q]["turns"] for q in sorted({i["qid"] for i in items})},
    "answers": {k: v for k, v in answers.items() if any(k == f"{i['qid']}|{i['a']}" or k == f"{i['qid']}|{i['b']}" for i in items)},
    "judgments": {f"{q}|{m}|{t}": judgment[(q, m, t)] for (q, m, t) in sorted(used)},
}
json.dump({"source": "LMSYS MT-Bench human judgments (CC BY 4.0) + GPT-4 single-answer grades", "items": items},
          open(f"{out}/mtbench_pairs.json", "w"), separators=(",", ":"))
json.dump(texts, open(f"{out}/texts.json", "w"), separators=(",", ":"))
print(len(items), "pairs;", sum(i["expert"] != "tie" for i in items), "non-tie;", len(texts["answers"]), "answers;", len(texts["judgments"]), "judgments")
