"""Local Korean tokenization diagnostics; these do not measure understanding."""
import argparse,json,random,hashlib,unicodedata
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer

PROBES={
 '조사와어미':['기관에 문의했어요.','기관에서 문의했어요.','기관으로 문의했어요.','기관이 문의했어요.','기관에 문의하지 않았어요.'],
 '띄어쓰기와구어':['출장 식비는 하루 얼마예요?','출장식비는 하루 얼마야?','하루 밥값 얼마까지 돼?','그건 어느 기관 기준인데?'],
 '단위와식별자':['식비 3만원, 정산 기한 7일, 보관 기간 3년입니다.','담당 부서는 정보지원팀이고 기관명은 제시되지 않았습니다.','서울 A기관의 IT지원팀에서 MacBook M5를 점검합니다.','물품 금액이 50만원 미만인지, 50만원 이상인지 확인하세요.'],
 '문맥과정정':['그게 무슨 뜻인가요?','제가 앞서 10일이라고 답한 것은 잘못입니다. 자료에는 7일이라고 되어 있습니다.','출장비 전체를 말씀하시는 건가요, 하루 식비를 말씀하시는 건가요?']}


def main():
 p=argparse.ArgumentParser();p.add_argument('--tokenizer',default='runs/trid-v0.2/tokenizer.json');p.add_argument('--data',default='data/wiki-ko-pilot/train.jsonl');p.add_argument('--sample',type=int,default=256);p.add_argument('--output',required=True)
 a=p.parse_args()
 if a.sample<1:p.error('sample must be positive')
 out=Path(a.output)
 if out.exists():raise ValueError('Use a new report path.')
 tok=Tokenizer.from_file(a.tokenizer);rng=random.Random(75);sample=[];count=0
 with Path(a.data).open() as f:
  for line in f:
   if not line.strip():continue
   r=json.loads(line);count+=1
   if len(sample)<a.sample:sample.append(r['text'])
   else:
    index=rng.randrange(count)
    if index<a.sample:sample[index]=r['text']
 total=Counter();roundtrip=0
 for text in sample:
  encoded=tok.encode(text)
  total['characters']+=len(text);total['nonspace_characters']+=sum(not c.isspace() for c in text);total['tokens']+=len(encoded.ids)
  total['hangul_characters']+=sum('가'<=c<='힣' or '\u1100'<=c<='\u11ff' or '\u3130'<=c<='\u318f' for c in text)
  roundtrip+=int(tok.decode(encoded.ids)==text)
 probes=[]
 for category,texts in PROBES.items():
  for text in texts:
   ids=tok.encode(text).ids;nfd=unicodedata.normalize('NFD',text)
   probes.append({'category':category,'text':text,'nfc_tokens':len(ids),'nfd_tokens':len(tok.encode(nfd).ids),'roundtrip':tok.decode(ids)==text})
 report={'tokenizer':a.tokenizer,'tokenizer_sha256':hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest(),'vocab_size':tok.get_vocab_size(),'source':a.data,'source_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'sampling':'reservoir, seed 75, full documents from existing training split only','source_documents':count,'sample_documents':len(sample),'counts':dict(total),'tokens_per_100_nonspace_characters':100*total['tokens']/total['nonspace_characters'],'sample_exact_roundtrips':roundtrip,'probes':probes,'limitations':'Tokenization efficiency and exact text reconstruction only; not Korean comprehension or institutional accuracy. Existing Wikipedia pilot is not a representative Korean language sample. Diagnostic prompts are development probes, not a hidden test set.'}
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v for k,v in report.items() if k!='probes'},ensure_ascii=False,indent=2))
 print('Probe NFC/NFD token totals:',sum(x['nfc_tokens'] for x in probes),sum(x['nfd_tokens'] for x in probes))

if __name__=='__main__':main()
