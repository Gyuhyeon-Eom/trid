"""Korean evidence-reading curriculum v2: diverse synthetic documents + KLUE MRC passages.

Changes from v1 (which had 4 fields, 10 amounts, 8 names, one fact per document):
- Values are sampled from wide generators, so answers must be copied from evidence, not recalled.
- Documents hold several fields in varied formats; questions must select one field (and one
  program qualifier when distractors share the field).
- Missing questions are contrastive: neighbouring fields or another program's value are present.
- Real Korean passages from KLUE MRC train (official dev excluded) in the same answer format.
Answer format is unchanged from v1: `<value> [Dn]`, `확인 불가`, `근거 충돌`.
"""
import argparse,hashlib,json,random,re,shutil,unicodedata
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
from trid.text import encode_example,SPECIAL
from trid.prompts import rag_prompt

V1=Path('data/korean-rag-curriculum-v1')
KLUE=Path('data/klue-mrc-source')
TOKENIZER='runs/trid-korean-medium-base-v1/tokenizer.json'
CONTEXT=1024
OUTPUT_RESERVE=64  # matches evaluation max_tokens
KINDS=['value','noise','table','missing','conflict','revision','combine']
SOURCE='authored synthetic evidence v2; fictional institutions, people and figures'

# ---------------------------------------------------------------- Korean helpers
def has_batchim(word):
    for ch in reversed(word):
        if '가'<=ch<='힣':return (ord(ch)-0xAC00)%28!=0
        if ch.isdigit():return ch in '013678'
        if ch.isalpha():return False
    return False

def josa(word,pair):
    a,b=pair.split('/');return word+(a if has_batchim(word) else b)

# ---------------------------------------------------------------- value generators
SURNAMES='김 이 박 최 정 강 조 윤 장 임 한 오 서 신 권 황 안 송 류 전 홍 고 문 양 손 배 백 허 유 남 심 노 하 곽 성 차 주 우 구 민 진 나 지 엄 채 원 천 방 공 현 함 변 염 여 추 도 소 석 선 설 마 길 연 위 표 명 기 반 왕 금 옥 육 인 맹 제 모 탁 국 어 은 편 용'.split()
GIVEN='가 경 계 고 관 광 교 구 국 규 균 근 기 길 나 남 다 단 담 대 덕 도 동 두 라 란 래 려 련 령 례 로 록 룡 류 륜 률 리 린 림 마 만 명 모 목 무 문 미 민 배 범 병 보 복 봉 빈 사 산 상 새 서 석 선 설 섭 성 세 소 솔 송 수 숙 순 슬 승 시 신 아 안 애 양 언 연 열 영 예 오 옥 완 요 용 우 욱 운 웅 원 위 유 윤 율 은 을 음 의 이 익 인 일 자 재 전 정 제 조 종 주 준 중 지 진 찬 창 채 천 철 초 춘 충 치 태 택 하 학 한 해 현 형 혜 호 홍 화 환 효 훈 휘 희'.split()
PLACES='한빛 새솔 늘봄 다온 푸른 미래 해오름 가람 누리 별빛 솔내 청송 은하 바른 동해 서진 남강 북한 금빛 하늘 온누리 새날 빛고을 이음 단비 고운 너른 맑은 들꽃 해든'.split()
REGIONS=['시','군','구','도']
ORG_TYPES='평생교육원 복지재단 문화재단 시설관리공단 인재개발원 산업진흥원 대학교 연구원 의료원 도서관 청소년수련관 체육회 신용보증재단 상권활성화재단 디자인센터 콘텐츠진흥원'.split()
DEPT_HEADS='재무 회계 인사 총무 기획 예산 시설 안전 정보보안 전산 교무 학사 학생지원 장학 연구지원 복지정책 민원 홍보 대외협력 감사 계약 구매 노무 교육운영 상생협력 시장지원 문화사업 고객지원 품질관리 경영지원'.split()
DEPT_TAILS=['팀','과','처','실','센터','부','지원단']
ROOMS=['대회의실','소회의실','세미나실','다목적홀','강의실','상담실','교육장','민원실','열람실','전시실']
BUILDINGS=['본관','별관','신관','제2청사','행정동','교육동','학생회관','복지관']

def person(r):return r.choice(SURNAMES)+r.choice(GIVEN)+r.choice(GIVEN)
def org(r):return r.choice(PLACES)+(r.choice(REGIONS)+' ' if r.random()<.5 else ' ')+r.choice(ORG_TYPES)
def dept(r):return r.choice(DEPT_HEADS)+r.choice(DEPT_TAILS)
def place(r):
    return r.choice([f'{r.choice(BUILDINGS)} {r.randint(1,9)}층 {r.choice(ROOMS)}',f'{r.choice(BUILDINGS)} {r.randint(1,6)}{r.randint(0,2)}{r.randint(1,9)}호',
                     f'{r.choice(PLACES)}홀',f'{r.choice(BUILDINGS)} {r.choice(ROOMS)}'])

def won(r):
    n=r.choice([r.randint(1,99)*1000,r.randint(1,300)*10000,r.randint(1,60)*5000,r.randint(10,990)*100,r.randint(1,50)*100000])
    style=r.random()
    if style<.45:return f'{n:,}원'
    if style<.7 and n%10000==0:return f'{n//10000}만'+r.choice(['원',' 원'])
    if style<.85 and n>=10000 and n%1000==0:
        man,chun=divmod(n//1000,10);return f'{man}만'+(f' {chun}천' if chun else '')+'원'
    return f'{n:,}원'
def money(r):
    v=won(r)
    if r.random()<.1:return f'미화 {r.randint(20,400)}달러'
    if r.random()<.15:return r.choice(['1일 ','월 ','1인당 ','1회 ','연 '])+v
    return v
def date(r):
    y,m,d=r.choice([2025,2026,2027]),r.randint(1,12),r.randint(1,28)
    wd='월화수목금토일'[r.randrange(7)]
    return r.choice([f'{y}년 {m}월 {d}일',f'{y}. {m}. {d}.',f'{m}월 {d}일',f'{m}월 {d}일({wd})',f'{y}-{m:02d}-{d:02d}'])
def clock(r):
    h,mi=r.randint(7,21),r.choice([0,0,10,20,30,30,40,50])
    if r.random()<.5:return f'{h:02d}:{mi:02d}'
    ap='오전' if h<12 else '오후';h12=h if h<=12 else h-12
    return f'{ap} {h12}시'+(f' {mi}분' if mi else '')
def period(r):
    n=r.randint(1,30)
    return r.choice([f'{n}일',f'{n}일 이내',f'{r.randint(1,24)}개월',f'{r.randint(1,10)}주',f'{n}일 전까지',f'{r.randint(1,5)}년'])
def percent(r):
    return r.choice([f'{r.randint(1,95)}%',f'{r.randint(1,30)}.{r.randint(1,9)}%',f'{r.randint(1,90)}퍼센트',f'{r.randint(5,95)}% 이내'])
COUNT_UNITS=[('일수','일'),('권수','권'),('인원','명'),('정원','명'),('선발','명'),('횟수','회'),('시간','시간'),('학점','학점'),('업체 수','곳')]
def count(r,unit='명'):
    n=r.randint(1,12) if unit in ('회','권') else r.randint(1,30) if unit in ('일','학점','곳') else r.randint(2,160) if unit=='시간' else r.randint(1,300)
    return f'{n}{unit}'+(' 이내' if r.random()<.1 else '')
def phone(r):
    area=r.choice(['02','031','032','042','044','051','053','061','063','064'])
    return r.choice([f'{area}-{r.randint(200,999)}-{r.randint(1000,9999)}',f'내선 {r.randint(1000,9999)}',f'1{r.choice("5866")}{r.randint(44,99)}-{r.randint(1000,9999)}'])
GEN={'money':money,'date':date,'time':clock,'period':period,'percent':percent,'count':count,'phone':phone,
     'person':person,'dept':dept,'place':place,'org':org}

# ---------------------------------------------------------------- domains: (label, kind, question aliases)
DOMAINS={
 '출장 여비':['출장 규정',['국내 출장','해외 출장','장기 출장','단기 연수'],[('일비','money',['일비','하루 일비']),('식비','money',['식비','하루 식비']),('숙박비 상한','money',['숙박비 상한','숙박비 한도']),('정산 기한','period',['정산 기한','정산은 언제까지']),('정산 담당','dept',['정산 담당 부서','정산하는 부서']),('교통비 기준','money',['교통비 기준','교통비']),('사전 승인권자','person',['사전 승인권자','승인하는 사람'])]],
 '휴가 복무':['복무 안내',['정규직','계약직','신규 임용자','육아기 직원'],[('연차 일수','count',['연차 일수','연차']),('병가 한도','period',['병가 한도','병가']),('신청 마감','period',['휴가 신청 마감','휴가 신청 기한']),('복무 담당','person',['복무 담당자','복무 담당']),('유연근무 시작 시각','time',['유연근무 시작 시각','유연근무는 몇 시부터']),('문의 전화','phone',['문의 전화','문의 번호'])]],
 '채용':['채용 공고',['일반직 신입','경력직','청년 인턴','연구직'],[('모집 인원','count',['모집 인원','뽑는 인원']),('접수 마감일','date',['접수 마감일','서류 마감일']),('면접 장소','place',['면접 장소','면접 보는 곳']),('월 급여','money',['월 급여','급여']),('채용 담당','dept',['채용 담당 부서','채용 문의처']),('합격자 발표일','date',['합격자 발표일','발표 날짜']),('가산점 비율','percent',['가산점 비율','가산점'])]],
 '장학금':['장학 안내',['성적 장학','가계 곤란 장학','근로 장학','대학원 연구 장학'],[('지급액','money',['지급액','장학금 액수']),('신청 기간 마감','date',['신청 마감일','신청 마감']),('선발 인원','count',['선발 인원','몇 명 뽑는지']),('최소 이수 학점','count',['최소 이수 학점','이수 학점 기준']),('담당 부서','dept',['담당 부서','담당하는 곳']),('담당자','person',['담당자','담당 직원'])]],
 '도서관':['이용 안내',['학부생','대학원생','교직원','지역 주민'],[('대출 권수','count',['대출 권수','빌릴 수 있는 책 수']),('대출 기간','period',['대출 기간','빌리는 기간']),('평일 운영 시작','time',['평일 운영 시작 시각','평일 개관 시간']),('연체료','money',['연체료','연체 시 요금']),('열람실 위치','place',['열람실 위치','열람실은 어디']),('문의 전화','phone',['문의 전화','도서관 전화번호'])]],
 '평생교육 강좌':['강좌 모집',['어르신 스마트폰반','직장인 영어반','어린이 코딩반','주부 요리반'],[('수강료','money',['수강료','수강 비용']),('정원','count',['정원','모집 정원']),('개강일','date',['개강일','개강 날짜']),('강의실','place',['강의실','수업 장소']),('강사','person',['강사','강사 이름']),('환불 기한','period',['환불 기한','환불 가능 기간'])]],
 '복지 지원금':['지원 사업 안내',['저소득 가구','한부모 가족','장애인 가구','청년 1인 가구'],[('지원 금액','money',['지원 금액','지원금']),('소득 기준','percent',['소득 기준','소득 요건']),('신청 기한','date',['신청 기한','신청 마감일']),('접수처','place',['접수처','신청하는 곳']),('담당 부서','dept',['담당 부서','담당 과']),('지급 횟수','count',['지급 횟수','지원 횟수'])]],
 '소상공인 보조금':['보조금 공고',['전통시장 점포','온라인 판로','시설 개선','경영 컨설팅'],[('업체당 지원 한도','money',['업체당 지원 한도','최대 지원액']),('자부담 비율','percent',['자부담 비율','자부담']),('지원 업체 수','count',['지원 업체 수','몇 곳 지원']),('공고 마감일','date',['공고 마감일','접수 마감']),('주관 기관','org',['주관 기관','주관하는 곳']),('상담 전화','phone',['상담 전화','상담 번호'])]],
 '시설 대관':['대관 안내',['평일 주간','평일 야간','주말','공휴일'],[('시간당 사용료','money',['시간당 사용료','대관료']),('최대 수용 인원','count',['최대 수용 인원','몇 명까지']),('예약 마감','period',['예약 마감','예약은 언제까지']),('대관 장소','place',['대관 장소','빌리는 장소']),('관리 책임자','person',['관리 책임자','시설 담당자']),('보증금','money',['보증금','예치금'])]],
 '주차 교통':['주차 안내',['방문객','정기권 이용자','경차','전기차'],[('최초 30분 요금','money',['최초 30분 요금','처음 30분 주차비']),('1일 최대 요금','money',['1일 최대 요금','하루 최대 주차비']),('할인율','percent',['할인율','할인']),('운영 종료 시각','time',['운영 종료 시각','몇 시까지 운영']),('주차장 위치','place',['주차장 위치','주차장은 어디']),('관리 부서','dept',['관리 부서','관리하는 부서'])]],
 '정보보안':['보안 지침',['일반 사용자','관리자 계정','외주 인력','원격 근무자'],[('비밀번호 변경 주기','period',['비밀번호 변경 주기','비밀번호 바꾸는 주기']),('로그인 실패 허용 횟수','count',['로그인 실패 허용 횟수','몇 번 틀리면 잠기는지']),('보안 교육 시간','count',['보안 교육 시간','보안 교육은 몇 시간']),('보안 담당','dept',['보안 담당 부서','보안 담당']),('계정 승인권자','person',['계정 승인권자','계정 승인하는 사람']),('침해 신고 전화','phone',['침해 신고 전화','사고 신고 번호'])]],
 '구매 계약':['계약 사무 처리 기준',['물품 구매','용역 계약','공사 계약','긴급 구매'],[('수의계약 한도','money',['수의계약 한도','수의계약 가능 금액']),('검수 기한','period',['검수 기한','검수 기간']),('계약 보증금 비율','percent',['계약 보증금 비율','계약 보증금']),('계약 담당','dept',['계약 담당 부서','계약 담당']),('대금 지급 기한','period',['대금 지급 기한','대금 지급']),('결재권자','person',['결재권자','최종 결재자'])]],
 '안전 점검':['안전 점검 계획',['전기 설비','소방 설비','승강기','가스 시설'],[('점검일','date',['점검일','점검 날짜']),('점검 시작 시각','time',['점검 시작 시각','점검 시작 시간']),('점검 주기','period',['점검 주기','몇 개월마다']),('점검 책임자','person',['점검 책임자','점검 담당자']),('점검 업체','org',['점검 업체','점검 기관']),('비상 연락처','phone',['비상 연락처','긴급 전화'])]],
 '사내 교육':['교육 운영 계획',['신입 사원','중간 관리자','임원','전 직원'],[('필수 이수 시간','count',['필수 이수 시간','몇 시간 들어야 하는지']),('교육 장소','place',['교육 장소','교육은 어디서']),('교육 시작일','date',['교육 시작일','교육 시작 날짜']),('1인당 교육비','money',['1인당 교육비','교육비']),('교육 담당','person',['교육 담당자','교육 담당']),('수료 기준 출석률','percent',['수료 기준 출석률','출석률 기준'])]],
 '외래 진료':['외래 안내',['내과','소아청소년과','정형외과','치과'],[('평일 접수 시작','time',['평일 접수 시작 시각','접수는 몇 시부터']),('토요일 진료 종료','time',['토요일 진료 종료 시각','토요일은 몇 시까지']),('초진 진찰료','money',['초진 진찰료','처음 진료비']),('예약 변경 기한','period',['예약 변경 기한','예약 변경은 언제까지']),('진료실 위치','place',['진료실 위치','진료실은 어디']),('예약 전화','phone',['예약 전화','예약 번호'])]],
 '행사 운영':['행사 안내',['개막식','체험 부스','공연','시상식'],[('행사일','date',['행사일','행사 날짜']),('시작 시각','time',['시작 시각','몇 시에 시작']),('참가비','money',['참가비','참가 비용']),('행사 장소','place',['행사 장소','어디서 열리는지']),('주최 기관','org',['주최 기관','주최하는 곳']),('참가 정원','count',['참가 정원','몇 명 참가'])]],
}
OLD_MARKS=['폐지된 규정','종전 기준','개정 전','구 규정','2024년 폐지','이전 안내(현재 효력 없음)','삭제된 조항']
NEW_MARKS=['현행 규정','개정 후','2026. 1. 1. 시행','최신 안내','현행 기준','시행 중인 개정 기준']
DOC_TYPES=['안내','공지','지침','기준','계획','안내문','운영 기준']
for _d in DOMAINS.values():  # count fields carry their unit: ('연차 일수','count:일',...)
    _d[2]=[(l,f'count:{next((u for k,u in COUNT_UNITS if k in l),"명")}' if k=='count' else k,a) for l,k,a in _d[2]]

PHRASE_MARKS=('몇','어디','언제','는지')
def label_question(r,alias,kind):
    a=alias;kind=kind.split(':')[0]
    if any(m in a for m in PHRASE_MARKS):  # aliases like '정산은 언제까지', '몇 명 뽑는지' are question phrases
        if a.endswith('는지'):return a+r.choice([' 알려주세요.',' 궁금해요.',' 알려줘.'])
        if a.endswith(('어디','까지','부터')):return a+r.choice(['야?','인가요?','예요?'])
        return a+(' ' if a.endswith(('마다','어디서')) else '')+r.choice(['해?','하나요?','하는지 알려줘.'])
    common=[f'{josa(a,"을/를")} 알려줘.',f'{josa(a,"을/를")} 알려주세요.',f'자료 기준으로 {josa(a,"은/는")} 어떻게 되나요?',f'{a} 확인 부탁드립니다.',f'{josa(a,"이/가")} 궁금합니다.',f'{josa(a,"은/는")}?']
    specific={'money':[f'{josa(a,"은/는")} 얼마인가요?',f'{josa(a,"이/가")} 얼마야?',f'{a} 금액이 어떻게 돼?'],
              'date':[f'{josa(a,"은/는")} 언제인가요?',f'{josa(a,"이/가")} 언제야?'],
              'time':[f'{josa(a,"은/는")} 몇 시예요?',f'{josa(a,"이/가")} 언제야?'],
              'period':[f'{josa(a,"은/는")} 얼마나 돼?',f'{josa(a,"은/는")} 어떻게 정해져 있나요?'],
              'percent':[f'{josa(a,"은/는")} 몇 퍼센트인가요?',f'{josa(a,"은/는")} 얼마야?'],
              'count':[f'{josa(a,"은/는")} 얼마나 되나요?',f'{josa(a,"이/가")} 몇이야?'],
              'phone':[f'{josa(a,"은/는")} 몇 번이에요?',f'{a} 번호 알려줘.'],
              'person':[f'{josa(a,"은/는")} 누구야?',f'{josa(a,"이/가")} 누구인가요?'],
              'dept':[f'{josa(a,"은/는")} 어디인가요?',f'{josa(a,"이/가")} 어디야?'],
              'place':[f'{josa(a,"은/는")} 어디예요?',f'{josa(a,"이/가")} 어디야?'],
              'org':[f'{josa(a,"은/는")} 어디인가요?',f'{josa(a,"이/가")} 어디야?']}[kind]
    if not a.endswith(('부서','장소','곳','위치','기관','업체','처')) and kind in ('dept','place','org'):specific=specific[:1]
    return r.choice(common+specific*2)

def render(r,title,facts,style=None):
    """facts: [(label,value)]. Returns passage text; every value appears verbatim."""
    style=style or r.choice(['sentence','bullet','article','table','faq','mail'])
    if style=='table':
        head=r.choice(['항목 | 내용','구분 | 기준','항목 | 값'])
        return title+'\n'+head+'\n'+'\n'.join(f'{l} | {v}' for l,v in facts)
    if style=='bullet':
        mark=r.choice(['-','■','○','·'])
        return title+'\n'+'\n'.join(f'{mark} {l}: {v}' for l,v in facts)
    if style=='article':
        start=r.randint(1,12)
        return title+'\n'+'\n'.join(f'제{start+i}조({l}) {josa(l,"은/는")} {v}이다.' for i,(l,v) in enumerate(facts))
    if style=='faq':
        return title+'\n'+'\n'.join(f'Q. {josa(l,"은/는")} 어떻게 되나요? A. {v}입니다.' for l,v in facts)
    if style=='mail':
        body=' '.join(f'{josa(l,"은/는")} {v}입니다.' for l,v in facts)
        return f'{title}\n안녕하세요. 문의하신 내용 안내드립니다. {body} 참고 바랍니다.'
    return title+' '+' '.join(f'{josa(l,"은/는")} {v}'+r.choice(['이다.','입니다.']) for l,v in facts)

def fresh(r,kind,avoid):
    base,_,unit=kind.partition(':')
    for _ in range(50):
        v=GEN[base](r,unit) if unit else GEN[base](r)
        if v not in avoid and not any(v in a or a in v for a in avoid):return v
    raise RuntimeError('value collision')

def make_doc(r,domain,qualifier,fields,values,mark=None,style=None):
    head,_,_=DOMAINS[domain]
    kind=r.choice(DOC_TYPES) if not head.endswith(('안내','공고','지침','기준','계획','모집')) else ''
    title=' '.join(x for x in [org(r) if r.random()<.5 else '',qualifier,head,kind] if x)
    if mark:title=f'{mark}: {title}' if r.random()<.5 else f'{title} - {mark}'
    return render(r,title,[(f[0],values[f[0]]) for f in fields],style)

def synthetic(r,kind,i,split):
    domain=r.choice(list(DOMAINS));_,quals,fields=DOMAINS[domain]
    ids=r.sample([f'D{k}' for k in range(1,10)],4)
    target_q,other_q=r.sample(quals,2)
    n=r.randint(3,min(5,len(fields)))
    chosen=r.sample(fields,n)
    target=chosen[0]
    used=set()
    def vals(fs):
        out={}
        for f in fs:out[f[0]]=fresh(r,f[1],used);used.add(out[f[0]])
        return out
    v1=vals(chosen)
    qualify=r.random()<.6
    def ask(field,q=None,force=False):
        alias=r.choice(field[2]);prefix=(q or target_q)+' ' if (force or qualify) else ''
        return label_question(r,prefix+alias,field[1])
    passages=[];answer=None
    def shuffle_cat(parts):
        r.shuffle(parts);return parts
    if kind=='value':
        passages=[(ids[0],make_doc(r,domain,target_q,shuffle_cat(list(chosen)),v1))]
        q=ask(target);answer=f'{v1[target[0]]} [{ids[0]}]'
    elif kind=='table':
        passages=[(ids[0],make_doc(r,domain,target_q,shuffle_cat(list(chosen)),v1,style='table'))]
        q=ask(target);answer=f'{v1[target[0]]} [{ids[0]}]'
    elif kind=='noise':
        v2=vals(chosen)  # same fields, different program: must match qualifier
        docs=[(ids[0],make_doc(r,domain,target_q,shuffle_cat(list(chosen)),v1)),(ids[1],make_doc(r,domain,other_q,shuffle_cat(list(chosen)),v2))]
        if r.random()<.5:
            d2=r.choice([d for d in DOMAINS if d!=domain]);f2=r.sample(DOMAINS[d2][2],3)
            docs.append((ids[2],make_doc(r,d2,r.choice(DOMAINS[d2][1]),f2,vals(f2))))
        passages=shuffle_cat(docs);q=ask(target,force=True);answer=f'{v1[target[0]]} [{ids[0]}]'
    elif kind=='missing':
        mode=r.random()
        if mode<.5:  # asked field absent, neighbours present
            absent=[f for f in fields if f not in chosen]
            if not absent:chosen=chosen[:-1];absent=[f for f in fields if f not in chosen]
            v1={f[0]:v1[f[0]] for f in chosen if f[0] in v1}
            passages=[(ids[0],make_doc(r,domain,target_q,shuffle_cat(list(chosen)),v1))]
            q=ask(r.choice(absent))
        else:  # field present only for another program
            passages=[(ids[0],make_doc(r,domain,other_q,shuffle_cat(list(chosen)),v1))]
            if r.random()<.5:
                d2=r.choice([d for d in DOMAINS if d!=domain]);f2=r.sample(DOMAINS[d2][2],3)
                passages.append((ids[1],make_doc(r,d2,r.choice(DOMAINS[d2][1]),f2,vals(f2))))
            q=ask(target,force=True)
        answer='확인 불가'
    elif kind=='conflict':
        alt={target[0]:fresh(r,target[1],used)};v2={**v1,**alt}
        rest=chosen[1:];r.shuffle(rest)
        passages=shuffle_cat([(ids[0],make_doc(r,domain,target_q,[target]+rest[:2],v1)),(ids[1],make_doc(r,domain,target_q,[target]+rest[2:3],v2))])
        q=ask(target);answer='근거 충돌'
    elif kind=='revision':
        alt={target[0]:fresh(r,target[1],used)};old={**v1,**alt}
        rest=chosen[1:]
        passages=shuffle_cat([(ids[0],make_doc(r,domain,target_q,[target]+rest[:2],v1,mark=r.choice(NEW_MARKS))),
                              (ids[1],make_doc(r,domain,target_q,[target]+rest[:2],old,mark=r.choice(OLD_MARKS)))])
        q=ask(target);answer=f'{v1[target[0]]} [{ids[0]}]'
    elif kind=='combine':
        d2=r.choice([d for d in DOMAINS if d!=domain]);f2=r.sample(DOMAINS[d2][2],3);v2=vals(f2);q2=r.choice(DOMAINS[d2][1])
        passages=shuffle_cat([(ids[0],make_doc(r,domain,target_q,shuffle_cat(list(chosen)),v1)),(ids[1],make_doc(r,d2,q2,f2,v2))])
        a1=f'{target_q} {target[0]}';a2=f'{q2} {f2[0][0]}'
        q=r.choice([f'{josa(a1,"과/와")} {josa(a2,"을/를")} 함께 알려줘.',f'{a1}, {a2} 각각 알려주세요.',f'{josa(a1,"과/와")} {josa(a2,"이/가")} 각각 어떻게 되나요?'])
        answer=f'{v1[target[0]]} [{ids[0]}], {v2[f2[0][0]]} [{ids[1]}]'
    return {'id':f'{split}-syn-{kind}-{i:05d}','prompt':rag_prompt(passages,q),'answer':answer,'task':'rag_'+kind,'group':'rag','domain':domain,'source':SOURCE}

# ---------------------------------------------------------------- KLUE passages
def norm(t):return unicodedata.normalize('NFC',t).strip()
def digest(t):return hashlib.sha256(re.sub(r'\s+','',norm(t)).encode()).hexdigest()
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def klue_rows(tok,r,limit_train,limit_valid,counts):
    prov=json.loads((KLUE/'provenance.json').read_text())
    for name,expected in prov['sha256'].items():
        if sha256(KLUE/name)!=expected:raise ValueError('KLUE source changed')
    train=json.loads((KLUE/'train.json').read_text())['data'];dev=json.loads((KLUE/'dev.json').read_text())['data']
    dev_titles={norm(x['title']) for x in dev};dev_ctx={digest(p['context']) for x in dev for p in x['paragraphs']}
    items={'train':[],'valid':[]};pool={'train':[],'valid':[]}
    for rec in train:
        title=norm(rec['title'])
        if title in dev_titles or any(digest(p['context']) in dev_ctx for p in rec['paragraphs']):
            counts['klue_excluded_dev_overlap']+=1;continue
        split='valid' if int(hashlib.sha256(title.encode()).hexdigest()[:8],16)%20==0 else 'train'
        for p in rec['paragraphs']:
            ctx=norm(p['context'])
            if '[D' in ctx or any(s in ctx for s in SPECIAL):counts['klue_skipped_markup']+=1;continue
            pool[split].append((title,ctx))
            for qa in p['qas']:items[split].append((title,ctx,p['context'],qa))
    out={'train':[],'valid':[]}
    for split,limit in [('train',limit_train),('valid',limit_valid)]:
        r.shuffle(items[split])
        for title,ctx,raw,qa in items[split]:
            if len(out[split])>=limit:break
            q=norm(qa['question'])
            if qa['is_impossible']:answer_text=None
            else:
                ok=[a for a in qa['answers'] if a.get('text') and raw[a['answer_start']:a['answer_start']+len(a['text'])]==a['text']]
                if not ok:counts['klue_invalid_span']+=1;continue
                answer_text=norm(ok[0]['text'])
                if answer_text not in ctx:counts['klue_answer_not_in_normalized']+=1;continue
            ids=r.sample([f'D{k}' for k in range(1,10)],2)
            passages=[(ids[0],ctx)]
            if r.random()<.5:
                other=r.choice(pool[split])
                if other[0]!=title:passages.append((ids[1],other[1]))
            def build(ps):
                r.shuffle(ps)
                answer='확인 불가' if answer_text is None else f'{answer_text} [{ids[0]}]'
                return {'id':f'{split}-klue-{qa["guid"]}','prompt':rag_prompt(ps,q),'answer':answer,
                        'task':'rag_klue_missing' if answer_text is None else 'rag_klue','group':'rag_real',
                        'source':'KLUE MRC v1.1 train (CC BY-SA 4.0), reformatted','source_guid':qa['guid'],'source_title':title}
            row=build(list(passages))
            if fits(tok,row):out[split].append(row);continue
            row=build(passages[:1])
            if fits(tok,row):out[split].append(row)
            else:counts['klue_too_long']+=1
    return out,prov

def fits(tok,row):
    try:ids,_=encode_example(tok,row,CONTEXT)
    except ValueError:return False
    prompt_len=len(ids)-len(tok.encode(row['answer']).ids)
    return prompt_len+OUTPUT_RESERVE<=CONTEXT

# ---------------------------------------------------------------- main
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/korean-rag-curriculum-v2')
    p.add_argument('--eval-source',default='data/korean-rag-eval-v1-source',help='authored held-out items; asserted absent from training')
    p.add_argument('--synthetic-per-kind',type=int,default=1600);p.add_argument('--klue-train',type=int,default=6000)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Use a new output directory')
    r=random.Random(271030);tok=Tokenizer.from_file(TOKENIZER);counts=Counter()
    rows={'train':[],'valid':[]};seen=set()
    for split,n in [('train',a.synthetic_per_kind),('valid',max(8,a.synthetic_per_kind//40))]:
        for kind in KINDS:
            made=0;tries=0
            while made<n:
                tries+=1
                if tries>n*5:raise RuntimeError(f'cannot generate {kind}')
                row=synthetic(r,kind,made,split)
                if row['prompt'] in seen:counts['synthetic_duplicate']+=1;continue
                if not fits(tok,row):counts['synthetic_too_long']+=1;continue
                seen.add(row['prompt']);rows[split].append(row);made+=1
    klue,klue_prov=klue_rows(tok,r,a.klue_train,max(100,a.klue_train//30),counts)
    for split in rows:rows[split]+=klue[split]
    # Base dialogue and non-RAG groups reused unchanged from v1 (weights recomputed below).
    for split in rows:
        for line in (V1/f'{split}.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['group']!='rag':row.pop('weight',None);rows[split].append(row)
    # Held-out evaluation must not leak: no evaluation passage or question text in training prompts.
    held=[json.loads(l) for f in sorted(Path(a.eval_source).glob('*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
    held_text={norm(e['text']) for h in held for e in h['evidence']}|{norm(h['question']) for h in held}
    for split,data in rows.items():
        for row in data:
            if any(t in row['prompt'] for t in held_text):raise ValueError(f'held-out text leaked into {split}: {row["id"]}')
    masses={'basic':.30,'public_work':.15,'limits':.10,'instruction':.05,'multi':.05,'rag':.23,'rag_real':.12}
    tasks=Counter((x['group'],x['task']) for x in rows['train']);groups=Counter(g for g,_ in tasks)
    for x in rows['train']:x['weight']=masses[x['group']]*len(rows['train'])/(groups[x['group']]*tasks[(x['group'],x['task'])])
    keys={s:{json.dumps([x.get('history',[]),x['prompt']],ensure_ascii=False,sort_keys=True) for x in d} for s,d in rows.items()}
    if keys['train']&keys['valid']:raise ValueError('train/valid prompt overlap')
    for s,d in rows.items():
        for x in d:
            if not fits(tok,x):raise ValueError(f'{s} row exceeds context: {x.get("id")}')
    out.mkdir(parents=True)
    for s,d in rows.items():
        r.shuffle(d);(out/f'{s}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in d))
    shutil.copyfile(V1/'base-check.jsonl',out/'base-check.jsonl')
    shutil.copyfile(KLUE/'License.md',out/'KLUE-License.md')
    (out/'ATTRIBUTION.txt').write_text('Rows with group rag_real derive from KLUE MRC v1.1 train. KLUE benchmark contributors. Source: https://github.com/KLUE-benchmark/KLUE\nRevision: '+klue_prov['revision']+'\nLicense: CC BY-SA 4.0 (KLUE-License.md). Changes: NFC, RAG prompt formatting, optional distractor passage from another title, answer reformatted as `<span> [Dn]` or 확인 불가, length filtering. Official dev excluded.\nNo commercial-clearance claim for every upstream document or resulting model.\n')
    provenance={'purpose':'Korean evidence-use curriculum v2 for medium SFT; mixed with v1 non-RAG dialogue rows',
      'changes_from_v1':'Wide value generators (copying instead of recall), 16 domains x 6-7 fields, multi-fact documents in 6 formats, program qualifiers with same-field distractors, contrastive missing questions, varied revision markers, KLUE real passages.',
      'tokenizer':TOKENIZER,'tokenizer_sha256':sha256(TOKENIZER),'context_length':CONTEXT,'output_reserve_tokens':OUTPUT_RESERVE,
      'sampling_mass':masses,'rows':{s:len(d) for s,d in rows.items()},
      'tasks':{s:dict(Counter(x['task'] for x in d)) for s,d in rows.items()},'counts':dict(counts),
      'held_out_check':f'No passage or question text from {a.eval_source} ({len(held)} items) occurs in any training/valid prompt.',
      'klue':{'source':klue_prov,'policy':'Official train only; titles/contexts overlapping official dev excluded; 1/20 title-hash to valid.'},
      'limitations':'Synthetic documents share generator structure; KLUE is Wikipedia/news reading, not institutional documents. Not a retrieval evaluation.',
      'hashes':{s:sha256(out/f'{s}.jsonl') for s in rows}}
    provenance['hashes']['base-check']=sha256(out/'base-check.jsonl')
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps({k:provenance[k] for k in ['rows','tasks','counts']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
