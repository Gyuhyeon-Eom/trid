"""Retrieval bottleneck check on data/korean-rag-eval-v1 (read-only).

Corpus: every distinct evidence passage (by text) across all 310 prompts. Queries: the '질문: ' line of each
answerable item (gold non-empty); relevant passages are the texts of that item's `sources`. Retrievers:
bm25_kiwi (bm25s + kiwipiepy morphemes, tags in KIWI_TAGS), bm25_space (whitespace control), kure (nlpai-lab/KURE-v1
dense, CPU, development measurement only, never shipped), hybrid (reciprocal rank fusion of bm25_kiwi + kure, k=60).
Metrics: Recall@1/3/5 (all relevant passages in top k), MRR (first relevant), mean/p50/p90 query latency on CPU.
"""
import argparse,hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from trid.prompts import rag_passages

KIWI_TAGS={'NNG','NNP','NNB','NR','NP','VV','VA','SN','SL','SH'}  # 명사(일반·고유·의존·수사·대명사), 동사·형용사 어간, 숫자, 알파벳, 한자
KS=(1,3,5);RRF_K=60
LICENCES={'bm25s':'MIT https://github.com/xhluca/bm25s','kiwipiepy':'Apache-2.0 https://github.com/bab2min/kiwipiepy',
  'sentence-transformers':'Apache-2.0 https://github.com/UKPLab/sentence-transformers','KURE-v1':'MIT https://huggingface.co/nlpai-lab/KURE-v1'}

def kiwi_tokenizer():
    from kiwipiepy import Kiwi
    kiwi=Kiwi()
    return lambda s:[t.form.lower() for t in kiwi.tokenize(s) if t.tag in KIWI_TAGS]

def bm25_retriever(docs,tokenize):
    import bm25s
    index=bm25s.BM25();index.index([tokenize(d) for d in docs],show_progress=False)
    def search(q,k):
        toks=tokenize(q)
        if not toks:return []
        scores=index.get_scores(toks);return [int(i) for i in np.argsort(-scores,kind='stable')[:k]]
    return search

def dense_retriever(docs,model_name,batch):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(model_name,device='cpu')
    t=time.perf_counter();emb=model.encode(docs,batch_size=batch,normalize_embeddings=True,show_progress_bar=False);index_s=time.perf_counter()-t
    def search(q,k):
        v=model.encode([q],normalize_embeddings=True,show_progress_bar=False)[0];return [int(i) for i in np.argsort(-(emb@v),kind='stable')[:k]]
    return search,index_s

def rrf(rankings,k):
    s=Counter()
    for r in rankings:
        for pos,d in enumerate(r):s[d]+=1/(RRF_K+pos+1)
    return [d for d,_ in s.most_common(k)]

def metrics(ranked,relevant):
    m={f'recall@{k}':relevant<=set(ranked[:k]) for k in KS}
    m['mrr']=next((1/(i+1) for i,d in enumerate(ranked) if d in relevant),0.0);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data',default='data/korean-rag-eval-v1')
    p.add_argument('--skip-dense',action='store_true',help='bm25 only (no KURE-v1 download/inference)')
    p.add_argument('--dense-model',default='nlpai-lab/KURE-v1');p.add_argument('--batch',type=int,default=8);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Use a new output file')
    root=Path(a.data);test=root/'test.jsonl';prov=json.loads((root/'provenance.json').read_text())
    digest=hashlib.sha256(test.read_bytes()).hexdigest()
    if digest!=prov['hashes']['test']:raise ValueError('Evaluation data changed')
    rows=[json.loads(l) for l in test.read_text().splitlines()]
    doc_id={};docs=[]
    for row in rows:
        for _,text in rag_passages(row['prompt']):
            if text not in doc_id:doc_id[text]=len(docs);docs.append(text)
    queries=[]
    for row in rows:
        if not row['gold']:continue
        by_src=dict(rag_passages(row['prompt']));q=row['prompt'].rsplit('\n질문: ',1)[1].strip()
        queries.append((row,q,{doc_id[by_src[s]] for s in row['sources']}))
    import torch;torch.set_num_threads(a.threads)
    kmax=max(KS);retrievers={};index_time={}
    t=time.perf_counter();retrievers['bm25_kiwi']=bm25_retriever(docs,kiwi_tokenizer());index_time['bm25_kiwi']=time.perf_counter()-t
    t=time.perf_counter();retrievers['bm25_space']=bm25_retriever(docs,lambda s:s.lower().split());index_time['bm25_space']=time.perf_counter()-t
    if not a.skip_dense:
        retrievers['kure'],index_time['kure']=dense_retriever(docs,a.dense_model,a.batch)
    names=list(retrievers)+(['hybrid'] if 'kure' in retrievers else [])
    per_query=[];agg={n:defaultdict(Counter) for n in names};lat={n:[] for n in names}
    for row,q,rel in queries:
        ranked={};rec={'id':row['id'],'origin':row['origin'],'task':row['task'],'relevant':sorted(rel),'top':{}}
        for n,search in retrievers.items():
            t=time.perf_counter();ranked[n]=search(q,kmax);lat[n].append(time.perf_counter()-t)
        if 'kure' in ranked:
            t=time.perf_counter();ranked['hybrid']=rrf([ranked['bm25_kiwi'],ranked['kure']],kmax);lat['hybrid'].append(time.perf_counter()-t+lat['bm25_kiwi'][-1]+lat['kure'][-1])
        for n in names:
            m=metrics(ranked[n],rel);rec['top'][n]=ranked[n][:kmax]
            for key in ('all',row['origin'],row['task']):
                c=agg[n][key];c['total']+=1
                for k,v in m.items():c[k]+=v
        per_query.append(rec)
    summary={n:{key:{**{m:round(c[m]/c['total'],4) for m in c if m!='total'},'total':c['total']} for key,c in sorted(g.items())} for n,g in agg.items()}
    latency={n:{'mean_ms':round(1e3*float(np.mean(v)),2),'p50_ms':round(1e3*float(np.median(v)),2),'p90_ms':round(1e3*float(np.percentile(v,90)),2)} for n,v in lat.items()}
    report={'data':a.data,'test_sha256':digest,'evaluated_at_unix':time.time(),'device':'cpu','threads':a.threads,'protocol':__doc__.strip(),
      'kiwi_tags':sorted(KIWI_TAGS),'rrf_k':RRF_K,'dense_model':None if a.skip_dense else a.dense_model,'dense_usage':'development measurement only; not part of the product',
      'licences':LICENCES if not a.skip_dense else {k:LICENCES[k] for k in ('bm25s','kiwipiepy')},
      'corpus_passages':len(docs),'queries':len(queries),'index_seconds':{k:round(v,2) for k,v in index_time.items()},
      'latency_per_query':latency,'summary':summary,'per_query':per_query}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=1))
    for n in names:
        s=summary[n]['all'];print(n,{m:s[m] for m in ('recall@1','recall@3','recall@5','mrr')},latency[n])

if __name__=='__main__':main()
