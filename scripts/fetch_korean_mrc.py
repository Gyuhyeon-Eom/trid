"""Download pinned public KLUE text; no weights, remote code, or uploads."""
import argparse
from pathlib import Path
from trid.corpus import download, write_json

PROVENANCE = {'repository': 'https://github.com/KLUE-benchmark/KLUE', 'revision': '3efd98708a40ff49251fddde35453f8fbb11f536', 'source_urls': {'train.json': 'https://raw.githubusercontent.com/KLUE-benchmark/KLUE/3efd98708a40ff49251fddde35453f8fbb11f536/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_train.json', 'dev.json': 'https://raw.githubusercontent.com/KLUE-benchmark/KLUE/3efd98708a40ff49251fddde35453f8fbb11f536/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_dev.json', 'License.md': 'https://raw.githubusercontent.com/KLUE-benchmark/KLUE/3efd98708a40ff49251fddde35453f8fbb11f536/License.md', 'README.md': 'https://raw.githubusercontent.com/KLUE-benchmark/KLUE/3efd98708a40ff49251fddde35453f8fbb11f536/README.md'}, 'sha256': {'train.json': 'c52c7b82a6015c09ea8be1fea49c912e73012d8b44d3653256d91c65d519c3df', 'dev.json': 'b31bb073e47cdeb19f2f1bae03d916ecabf945140334c75842d6994f700d4f47', 'License.md': '7abe19ec9bb73b36141b999b861d24ad855e808bafe0f81e84cce28556f6c297', 'README.md': '1f87facecc35feabf8ad25ecd2eda75bab8ea3ad074569b5239b4d57197ea059'}, 'purpose': 'Public Korean MRC data preparation; not model weights. Official dev is excluded from training.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/klue-mrc-source')
    a=p.parse_args(); root=Path(a.output)
    for name,url in PROVENANCE['source_urls'].items():
        download(url,root/name,PROVENANCE['sha256'][name])
        print(name,flush=True)
    write_json(root/'provenance.json',PROVENANCE)

if __name__=='__main__':main()
