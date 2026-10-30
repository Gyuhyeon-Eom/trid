"""Official abbreviations of central administrative bodies from 「정부조직 약칭과 영어 명칭에 관한 규칙」 [별표].

Each version's annex PDF is downloaded from law.go.kr (the file its version page references) and kept with its sha256,
so every alias is traceable to the official text. Rows are parsed as "번호 기관명 약칭 영어 명칭 영어 약칭"; a row whose
abbreviation equals its name adds nothing. The current version's rows are `current`; a superseded version adds only
abbreviations the current version no longer uses, mapped to the body that version named (고용부 → 고용노동부,
기재부 → 기획재정부), marked `superseded`.

    .venv/bin/python scripts/build_official_abbreviations.py
"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/official-abbreviations'
RULE = '정부조직 약칭과 영어 명칭에 관한 규칙'
ANNEX = '[별표] 중앙행정기관의 약칭과 영어 명칭(제14조 관련)'
VERSIONS = [  # newest first; the first is the version in force
    {'id': 'mois-yegyu-390', 'version': '행정안전부예규 제390호, 2026. 9. 30., 일부개정 (시행 2026. 10. 2.)',
     'version_page': 'https://www.law.go.kr/LSW//admRulInfoP.do?admRulSeq=2100000285900&chrClsCd=010201',
     'annex_url': 'https://www.law.go.kr/LSW/flDownload.do?bylClsCd=200201&flSeq=170019313',
     'annex_sha256': '67ff964ff0d03cdcbd9b18c357005eeffd67275c936e1cd38c87456c5c1129de'},
    {'id': 'mois-yegyu-269', 'version': '행정안전부예규 제269호, 2023. 12. 11., 일부개정 (시행 2023. 12. 11.)',
     'version_page': 'https://www.law.go.kr/LSW/admRulInfoP.do?admRulSeq=2100000232398&chrClsCd=010201',
     'annex_url': 'https://www.law.go.kr/LSW/flDownload.do?bylClsCd=200201&flSeq=135326009',
     'annex_sha256': '909f0a9a2cd376fa73ba543e555523e95c850da629febabe6e14cd5112ad875f'},
]
ROW = re.compile(r'^(\d+) (\S+) (\S+)(?: |$)')


def parse(pdf):
    import pypdf

    rows, numbers, names = [], [], []
    for page in pypdf.PdfReader(pdf).pages:
        for line in page.extract_text().splitlines():
            m = ROW.match(line.strip())
            if m:
                numbers.append(int(m[1]))
                names.append(m[2])
                if m[2] != m[3]:
                    rows.append({'alias': m[3], 'canonical': m[2], 'number': int(m[1])})
    if numbers != list(range(1, len(numbers) + 1)):
        raise SystemExit(f'{pdf}: annex rows are not numbered 1..n: {numbers}')
    return rows, names


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, sources = [], []
    for i, v in enumerate(VERSIONS):
        pdf = OUT / f"{v['id']}-annex.pdf"
        if not pdf.exists():
            pdf.write_bytes(urllib.request.urlopen(v['annex_url'], timeout=60).read())
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != v['annex_sha256']:
            raise SystemExit(f"{pdf}: sha256 differs from the pinned {v['id']} annex; review the new file before using it")
        rows, names = parse(pdf)
        bodies = len(names)
        if i == 0:
            central = names  # 중앙행정기관 listed by the version in force: their staff are 국가공무원
        status = 'current' if i == 0 else 'superseded'
        known = {r['alias'] for r in table}
        table += [{**r, 'status': status, 'version': v['id']} for r in rows if r['alias'] not in known]
        sources.append({**v, 'status': status, 'annex_file': pdf.name, 'bodies': bodies})
    path = OUT / 'abbreviations.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in table))
    manifest = {'rule': RULE, 'annex': ANNEX, 'versions': sources, 'abbreviations': len(table),
                'table_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'central_bodies': central}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
