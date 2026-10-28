"""Build a traceable short-dialogue set; evaluation prompts are excluded."""
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
import argparse

from tokenizers import Tokenizer
from trid.text import encode_example

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/dialogue-v1"


def normalize(text):
    return re.sub(r"[^가-힣a-z0-9]", "", unicodedata.normalize("NFC", text).lower())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='data/dialogue-v1')
    parser.add_argument('--expanded',action='store_true')
    parser.add_argument('--multi-turn',action='store_true')
    args=parser.parse_args()
    OUT=ROOT/args.output
    if OUT.exists():
        raise SystemExit("Output exists; preserve it and use a new dataset version.")
    OUT.mkdir(parents=True)
    tok = Tokenizer.from_file(str(ROOT / "data/wiki-ko-tokens/tokenizer.json"))
    heldout = set()
    for file in ["dialogue-eval-dev.jsonl", "dialogue-eval-final.jsonl"]:
        for line in (ROOT / "data" / file).read_text().splitlines():
            heldout.add(normalize(json.loads(line)["prompt"]))
    splits = {s: [] for s in ["train", "valid", "test"]}
    seen = set()
    excluded = 0

    def add(row, split):
        nonlocal excluded
        key = json.dumps([row.get("history", []), normalize(row["prompt"])], ensure_ascii=False, sort_keys=True)
        if normalize(row["prompt"]) in heldout or key in seen:
            excluded += 1
            return
        try:
            encode_example(tok, row, 256)
        except ValueError:
            excluded += 1
            return
        seen.add(key)
        splits[split].append(row)

    intents = json.loads((ROOT / "data/dialogue-intents.json").read_text())
    expansion=json.loads((ROOT/'data/dialogue-expansion.json').read_text()) if args.expanded else {}
    reserved={normalize(q) for intent in intents for q in intent['questions']+expansion.get(intent['topic'],[])}
    banned = re.compile("자살|죽고|죽을|살인|폭행|마약|도박|주식|비트코인|가상화폐|가스불|성관계|임신|낙태|치료|진단|처방|약을|병원|암에|종교|대통령")
    with (ROOT / "data/dialogue-source/ChatbotData.csv").open() as f:
        for r in csv.DictReader(f):
            q, a = r["Q"].strip(), r["A"].strip()
            if r["label"].strip() != "0" or len(q) > 100 or not 4 <= len(a) <= 120 or banned.search(q + a) or (args.expanded and normalize(q) in reserved):
                continue
            # Group identical answers so common paraphrases are kept together.
            bucket = int(digest(normalize(a))[:8], 16) % 100
            split = "test" if bucket < 5 else "valid" if bucket < 10 else "train"
            add({"prompt": q, "answer": a, "task": "public_daily", "weight": 1,
                 "source": "songys/Chatbot_data", "source_license": "MIT"}, split)
    for intent in intents:
        for i, question in enumerate(intent["questions"]):
            split = "valid" if i == len(intent["questions"]) - 1 else "train"
            for j, prefix in enumerate(["", "저기, ", "있잖아, "] if split == "train" else [""]):
                add({"prompt": prefix + question, "answer": intent["answers"][(i + j) % len(intent["answers"])],
                     "task": intent["topic"], "weight": 8, "source": "authored Korean dialogue curriculum"}, split)
        for i, question in enumerate(expansion.get(intent['topic'],[])):
            split='valid' if i==len(expansion[intent['topic']])-1 else 'train'
            for j,prefix in enumerate(['','저기요, ','음, ','그런데 '] if split=='train' else ['']):
                add({'prompt':prefix+question,'answer':intent['answers'][(i+j)%len(intent['answers'])],
                     'task':intent['topic'],'weight':4,'source':'authored diverse dialogue phrasings'},split)
    names = ["가람", "나래", "다온", "라온", "마루", "민수", "지훈", "지민", "하늘", "수빈", "유진", "지원", "도윤", "예린", "수현", "지우"]
    recall_questions = ["내 이름이 뭐야?", "아까 말한 이름을 알려줘", "내 이름 기억해?", "내 이름을 다시 말해 줘"]
    for i, name in enumerate(names):
        for intro in [f"내 이름은 {name}야.", f"나는 {name}라고 해.", f"{name}라고 불러 줘."]:
            if args.expanded:
                add({'prompt':intro,'answer':f'반가워요, {name}님. 저는 trid입니다.',
                     'task':'name_introduction','weight':8,'source':'authored contextual dialogue'},'train')
            history = [{"user": intro, "assistant": f"반가워요, {name}님."}]
            for j, q in enumerate(recall_questions):
                add({"history": history, "prompt": q, "answer": f"이름은 {name}라고 말씀하셨어요.",
                     "task": "name_memory", "source": "authored contextual dialogue", "weight": 8},
                    "valid" if j == 3 else "train")
    followups = [
        ("오늘 너무 피곤해.", "잠깐 쉬면서 숨을 돌려 보세요.", ["응 좀 쉴게", "잠깐 쉬고 올게", "조금 쉬어야겠다"], "좋아요. 편하게 쉬고 오세요."),
        ("오늘 할 일이 많아.", "가장 급한 일부터 하나씩 정리해 보세요.", ["하나씩 해 볼게", "급한 것부터 해야겠다", "먼저 목록을 적어 볼게"], "좋아요. 하나씩 차근차근 해보세요."),
        ("점심 뭐 먹지?", "비빔밥이나 샌드위치는 어떠세요?", ["비빔밥 먹을래", "샌드위치로 정했어", "추천한 메뉴로 할게"], "좋아요. 맛있게 드세요!"),
        ("오늘 생일이야.", "생일 축하해요! 좋은 하루 보내세요.", ["축하해 줘서 고마워", "고마워 기분 좋네", "감사해"], "천만에요. 즐거운 생일 보내세요!"),
        ("네 이름이 뭐야?", "저는 작은 언어모델 trid입니다.", ["trid 반가워", "앞으로 잘 부탁해", "만나서 반갑네"], "저도 반가워요! 편하게 이야기해 주세요."),
        ("집중이 안 돼.", "작은 일 하나부터 짧게 시작해 보세요.", ["그렇게 해 볼게", "알았어 시작해 볼게", "좋은 생각이네"], "좋아요. 부담 갖지 말고 조금씩 시작해 보세요.")
    ]
    for q, a, replies, response in followups:
        for i, reply in enumerate(replies):
            for prefix in ["", "응, ", "그래, "]:
                add({"history": [{"user": q, "assistant": a}], "prompt": prefix + reply,
                     "answer": response, "task": "followup", "source": "authored contextual dialogue", "weight": 8},
                    "valid" if i == 2 else "train")
    if args.multi_turn:
        rng=random.Random(73)
        prior_pool=[r for r in splits['train'] if not r.get('history') and r['task']!='public_daily']
        for split in ['train','valid']:
            for row in list(splits[split]):
                if row.get('history') or row['task']=='public_daily':
                    continue
                prior=rng.choice(prior_pool)
                history=[{'user':prior['prompt'],'assistant':prior['answer']}]
                add({**row,'history':history,'source':'authored dialogue with independently sampled previous turn'},split)
                if split=='train':
                    add({**row,'history':[{'user':'안녕','assistant':'안녕하세요! 편하게 이야기해 주세요.'}],
                         'source':'authored dialogue following greeting'},split)
    for split, rows in splits.items():
        random.Random(42).shuffle(rows)
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    source = json.loads((ROOT / "data/dialogue-source/source.json").read_text())
    provenance = {"source": source, "authored_data": "data/dialogue-intents.json and this builder; generated by assistant, no real company data",
                  "counts": {s: len(v) for s,v in splits.items()}, "excluded_or_duplicate": excluded,
                  "sampling_weights": "public=1, authored core=8, expanded phrasings=4; curriculum topics deliberately oversampled",
                  "multi_turn": args.multi_turn,
                  "split": "public grouped by normalized answer; authored held-out wording; final eval excluded by normalized prompt",
                  "limitations": "Some public near-duplicates may remain. Authored validation shares intents with training. Do not infer general conversation ability from loss alone.",
                  "sha256": {s: hashlib.sha256((OUT/f'{s}.jsonl').read_bytes()).hexdigest() for s in splits}}
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps(provenance,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
