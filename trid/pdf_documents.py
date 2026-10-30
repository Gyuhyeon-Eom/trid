"""Local PDF text ingestion and evidence retrieval; never infers an institution from filenames."""
from dataclasses import dataclass
from collections import Counter
from io import BytesIO
import math,re,unicodedata
from pypdf import PdfReader

MAX_BYTES=20*1024*1024
MAX_PAGES=80
MAX_CHARACTERS=300_000


class DocumentError(ValueError):pass


@dataclass(frozen=True)
class Chunk:
    page:int
    start:int
    end:int
    text:str


@dataclass
class Document:
    name:str
    pages:list[str]
    chunks:list[Chunk]
    warnings:list[str]


def extract_pdf(data,name):
    if len(data)>MAX_BYTES:raise DocumentError('PDF는 20MB 이하로 넣어 주세요.')
    if not data.startswith(b'%PDF-'):raise DocumentError('PDF 파일 형식이 아닙니다.')
    try:
        reader=PdfReader(BytesIO(data),strict=False)
        if reader.is_encrypted:raise DocumentError('암호화된 PDF입니다. 로컬에서 암호를 해제한 사본을 넣어 주세요.')
        if not 1<=len(reader.pages)<=MAX_PAGES:raise DocumentError('PDF는 1~80페이지까지 지원합니다.')
        pages=[];warnings=[];count=0
        for i,page in enumerate(reader.pages,1):
            text=page.extract_text(extraction_mode='layout') or ''
            text=unicodedata.normalize('NFC',text).replace('\x00','')
            text='\n'.join(re.sub(r'[ \t]+',' ',line).strip() for line in text.splitlines())
            text=re.sub(r'\n{3,}','\n\n',text).strip()
            count+=len(text)
            if count>MAX_CHARACTERS:raise DocumentError('추출된 글자가 너무 많습니다. 필요한 페이지를 분리해서 넣어 주세요.')
            pages.append(text)
            if len(re.sub(r'\s','',text))<10:warnings.append(f'{i}페이지: 글자가 거의 추출되지 않았습니다. 스캔·이미지 또는 빈 페이지인지 확인해 주세요.')
        if not any(len(re.sub(r'\s','',p))>=10 for p in pages):
            raise DocumentError('읽을 수 있는 텍스트가 없습니다. 이미지·스캔 PDF는 OCR로 텍스트를 만든 뒤 넣어 주세요. 현재 앱은 OCR을 하지 않습니다.')
        warnings.append('표·다단 문서의 읽기 순서는 달라질 수 있습니다. 추출 문서 보기에서 확인해 주세요.')
        return pages,warnings
    except DocumentError:raise
    except Exception as error:
        raise DocumentError('PDF를 읽지 못했습니다. 손상 여부 또는 텍스트 추출 가능 여부를 확인해 주세요.') from error


def make_chunks(pages,tok,budget=128,overlap=32):
    if not 0<=overlap<budget:raise ValueError('Invalid chunk window')
    chunks=[]
    for page,text in enumerate(pages,1):
        if not text.strip():continue
        encoded=tok.encode(text);offsets=encoded.offsets
        previous=None
        for pos in range(0,len(offsets),budget-overlap):
            window=offsets[pos:pos+budget]
            if not window:continue
            start=min(a for a,b in window);end=max(b for a,b in window)
            # Prefer complete source lines when all fit; otherwise retain an overlapping token window.
            key=(start,end)
            if key==previous:continue
            previous=key;part=text[start:end]
            if part.strip():chunks.append(Chunk(page,start,end,part))
            if pos+budget>=len(offsets):break
    if not chunks:raise DocumentError('질문할 수 있는 문서 구간이 없습니다.')
    return chunks


def load_pdf(data,name,tok):
    # Browser upload name is a display label only, never a filesystem path or source fact.
    name=name.replace('\\','/').rsplit('/',1)[-1][:180] or 'document.pdf'
    pages,warnings=extract_pdf(data,name)
    return Document(name,pages,make_chunks(pages,tok),warnings)


def terms(text):
    compact=re.sub(r'[^가-힣A-Za-z0-9]','',unicodedata.normalize('NFC',text)).lower()
    return Counter(compact[i:i+2] for i in range(max(0,len(compact)-1)))


def retrieve(document,question,limit=6):
    query=terms(question)
    all_terms=[terms(c.text) for c in document.chunks]
    df=Counter(t for ts in all_terms for t in ts)
    n=len(all_terms);ranked=[]
    for i,ts in enumerate(all_terms):
        score=sum((1+math.log(1+n/(1+df[t])))*min(count,ts[t]) for t,count in query.items() if t in ts)
        if score>0:ranked.append((score,i))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    return [(document.chunks[i],score) for score,i in ranked[:limit]]


def answer_document(document,question,span,presence,tok,device,reader=None,selected=None):
    from .short_reader import read
    if reader is None:reader=read
    question=question.strip()
    if not question:raise DocumentError('질문을 입력해 주세요.')
    if len(tok.encode(question).ids)>96:raise DocumentError('질문을 조금 짧게 입력해 주세요. 질문 한도는 96토큰입니다.')
    # `selected`: [(chunk, score)] already retrieved elsewhere (e.g. a shared search index); chunks need text/page/start.
    if selected is None:selected=retrieve(document,question)
    checked=[];candidates=[]
    for chunk,relevance in selected:
        r=reader(span,presence,tok,chunk.text,question,device)
        item={'page':chunk.page,'text':chunk.text,'page_text_start':chunk.start,'retrieval_score':round(relevance,3),
              'prediction':r}
        checked.append(item)
        if not r['abstains']:
            start,end=r['source_character_span']
            if not 0<=start<end<=len(chunk.text) or chunk.text[start:end]!=r['answer']:
                raise DocumentError('모델의 원문 위치 검증에 실패했습니다.')
            item['page_character_span']=[chunk.start+start,chunk.start+end]
            candidates.append(item)
    # Different predicted spans are shown as alternatives, not silently called a policy conflict.
    answers={re.sub(r'\s+','',c['prediction']['answer']).strip('.'):c['prediction']['answer'] for c in candidates}
    if not selected:
        status='not_found';answer='질문과 관련된 문서 구간을 찾지 못했습니다. 질문을 바꾸거나 추출 내용을 확인해 주세요.'
    elif not candidates:
        status='not_found';answer='확인한 문서 구간에서 답을 찾지 못했습니다.'
    elif len(answers)>1:
        status='review';answer='모델이 서로 다른 답 후보를 골랐습니다. 아래 원문을 확인해 주세요.'
    else:
        status='answer';answer=next(iter(answers.values()))
    return {'status':status,'answer':answer,'file':document.name,'candidates':candidates,'checked':checked,
            'checked_chunks':len(checked),'total_chunks':len(document.chunks),
            'scope':'관련도 상위 최대 6개 구간을 확인한 결과입니다. 미검색 구간까지 정보가 없다고 보장하지 않습니다.',
            'warnings':document.warnings}
