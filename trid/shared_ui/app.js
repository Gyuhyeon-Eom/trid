'use strict';
const el = id => document.getElementById(id);
const token = document.querySelector('meta[name="trid-token"]').content;
let documentId = null, pages = [], documentName = '', busy = false, lastApplied = null;
let commandHistory = [], historyIndex = 0, draft = '';
function print(text, kind = '') {
  const line = document.createElement('pre');
  line.className = kind;
  line.textContent = text;
  el('output').append(line);
  line.scrollIntoView({block:'end'});
}
function setBusy(value) {
  busy = value;
  el('command').disabled = value;
  el('command-form').querySelector('button').disabled = value;
  el('save-text').disabled = value;
  el('cancel-text').disabled = value;
  el('version').disabled = value;
  el('pending').textContent = value ? 'trid > 처리 중…' : '';
  if (!value && el('paste-panel').hidden) el('command').focus();
}
async function api(path, body, headers = {}) {
  const response = await fetch(path, {method:'POST', headers:{'X-Trid-Token':token, ...headers}, body});
  let data;
  try {data = await response.json();} catch {throw Error('서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.');}
  if (!response.ok) throw Error(data.error || '요청에 실패했습니다.');
  return data;
}
const jsonApi = (path, data) => api(path, JSON.stringify(data), {'Content-Type':'application/json'});
async function task(action) {
  if (busy) return;
  setBusy(true);
  try {await action();} catch (error) {print(`오류 > ${error.message}`, 'error');} finally {setBusy(false);}
}
function loaded(data) {
  documentId = data.document_id; pages = data.pages; documentName = data.name;
  print(`trid > ${documentName} · ${pages.length}페이지를 읽었습니다.\n이제 질문은 이 문서에서만 찾습니다. 공유 자료 검색으로 돌아가려면 /workspace.`, 'reply');
  for (const warning of data.warnings) print(warning.replace('추출 문서 보기에서 확인해 주세요.', '/source로 원문을 확인해 주세요.'), 'muted');
}
function help() {
  print('말을 걸거나 질문하세요. 업무 규정·기준 질문은 등록된 공개 자료에서 근거를 찾습니다.\n/doc 번호  (고급) 직전 답변의 참고 문서 중 하나로 다음 질문 고정, /doc 해제\n/workspace 업로드 문서 대신 공유 자료 검색으로 돌아가기\n/pdf       텍스트 PDF 열기 (2MB · 20페이지 · 5만 자 이하, OCR 미지원)\n/paste     문서 직접 붙여넣기 (5,000자 이하)\n/sample    가상 출장 규정 불러오기\n/source    현재 문서(또는 직전 답변의 적용 문서 원문 링크) 보기\n/clear     대화 기억·서버 문서·화면 기록 지우기\n/help      도움말\n\n기본 엔진은 최근 대화 몇 턴을 기억합니다(이 브라우저 세션, /clear로 삭제). 업무 사실은 매번 공개 자료에서 근거를 새로 찾습니다.\n↑ / ↓ 이전 명령 선택 · Enter 실행', 'muted');
}
function describe(d) {
  return [d.institution, d.title, d.kind, d.date ? `${d.date_type || '기준일'} ${d.date}` : null, d.effective_date ? `시행 ${d.effective_date}` : null].filter(Boolean).join(' · ');
}
function pastePanel(show) {
  el('paste-panel').hidden = !show;
  el('command-form').hidden = show;
  (show ? el('document-text') : el('command')).focus();
}
async function execute(value) {
  print(`나 > ${value}`);
  if (value === '/help') {help(); return;}
  if (value === '/pdf') {el('file').value = ''; el('file').click(); return;}
  if (value === '/paste') {pastePanel(true); return;}
  if (value === '/source') {
    if (documentId) print(`${documentName}\n\n${pages.map((p,i)=>`[${i+1}페이지]\n${p}`).join('\n\n')}`, 'muted');
    else if (lastApplied) print(`직전 답변의 적용 문서: ${describe(lastApplied)}\n원문: ${lastApplied.source}`, 'muted');
    else print('trid > 먼저 질문하거나 /pdf, /paste로 문서를 넣어 주세요.', 'muted');
    return;
  }
  if (value === '/workspace') {documentId = null; pages = []; documentName = ''; print('trid > 공유 자료 검색으로 돌아왔습니다.', 'muted'); return;}
  if (value === '/doc' || /^\/doc\s+\d+$/.test(value)) {
    const n = value === '/doc' ? null : Number(value.split(/\s+/)[1]);
    await task(async()=>{
      const r = await jsonApi('/api/pin', {choice:n});
      print(r.pinned ? `trid > 다음 질문부터 이 문서에서만 찾습니다: ${describe(r.pinned)}` : 'trid > 문서 고정을 풀었습니다. 질문마다 가장 관련 있는 문서를 자동으로 고릅니다.', 'muted');
    }); return;
  }
  if (value === '/sample') {await task(async()=>loaded(await jsonApi('/api/sample', {}))); return;}
  if (value === '/clear' || value === '/reset') {
    await task(async()=>{
      await jsonApi('/api/clear', {});
      documentId = null; pages = []; documentName = ''; commandHistory = []; historyIndex = 0; draft = '';
      el('file').value = ''; el('document-text').value = ''; el('output').replaceChildren();
      print('TRID > 대화 기억, 문서, 화면 기록을 지웠습니다.');
    }); return;
  }
  if (value.startsWith('/')) {print('trid > 알 수 없는 명령입니다. /help로 명령어를 확인하세요.', 'muted'); return;}
  await task(async()=>{
    const started = performance.now();
    const request = documentId ? {question:value, document_id:documentId, version:el('version').value}
                               : {question:value, scope:'workspace', version:el('version').value};
    const result = await jsonApi('/api/ask', request);
    if (result.intent === 'chat') {
      print(`TRID > ${result.answer}`, 'reply');
      print(`[일반 대화 · 문서 근거 없음 · 검증 대상 아님 · ${result.model} · ${result.elapsed_ms}ms]`, 'muted');
      for (const note of result.commentary_notes || []) print(note, 'error');
      return;
    }
    print(`TRID [근거 검색 · ${result.model}] > ${result.answer}`, 'reply');
    if (result.search_query) print(`[앞선 대화를 반영한 검색어] ${result.search_query}`, 'muted');
    const ws = result.workspace;
    if (ws) {
      lastApplied = ws.applied;
      if (ws.applied) print(`[적용 문서${ws.pinned ? ' · 고정됨' : ''}] ${describe(ws.applied)}\n원문: ${ws.applied.source} · ${ws.applied.license}\n${ws.applied.currency || ''}`, 'muted');
    }
    print(`[서버 처리 ${result.elapsed_ms}ms · 왕복 ${Math.round(performance.now()-started)}ms · ${result.checked_chunks}/${result.total_chunks}개 구간 확인]`, 'muted');
    for (const candidate of result.candidates) {
      const [a,b] = candidate.prediction.source_character_span;
      const chars = Array.from(candidate.text);
      // Code-point offsets are returned by the Python model; bracket the exact source span.
      const source = chars.slice(0,a).join('') + '【' + chars.slice(a,b).join('') + '】' + chars.slice(b).join('');
      const where = candidate.source_label || `${candidate.page}페이지`;
      print(`[${result.answer_mode === 'case' ? '원문 인용 · 글자 그대로 확인됨' : '근거'} · ${where}]\n${source}`, 'muted');
    }
    if (result.commentary) print(`[모델 해설 · 검증 안 됨 — 원문과 대조하세요]\n${result.commentary}`, 'muted');
    for (const note of result.commentary_notes || []) print(note, note.includes('숨겼습니다') ? 'error' : 'muted');
    if (result.table_warning) print(result.table_warning, 'error');
    if (ws?.selection === 'answer' && ws.others?.length) print('검색된 다른 문서 (참고용, 이 답변에는 쓰지 않음. 다른 기관 기준이 필요하면 질문에 기관명이나 규정명을 넣어 주세요):\n' + ws.others.slice(0, 3).map(d=>`- ${describe(d)}`).join('\n'), 'muted');
    print(result.scope, 'muted');
  });
}
el('command-form').addEventListener('submit', event=>{
  event.preventDefault();
  const value = el('command').value.trim();
  if (busy || !value) return;
  commandHistory.push(value); historyIndex = commandHistory.length; draft = '';
  el('command').value = '';
  void execute(value);
});
el('command').addEventListener('keydown', event=>{
  if (event.key === 'Enter' && event.isComposing) {event.preventDefault(); return;}
  if (event.isComposing || !['ArrowUp','ArrowDown'].includes(event.key) || !commandHistory.length) return;
  event.preventDefault();
  if (historyIndex === commandHistory.length) draft = el('command').value;
  historyIndex = Math.max(0, Math.min(commandHistory.length, historyIndex + (event.key === 'ArrowUp' ? -1 : 1)));
  el('command').value = historyIndex === commandHistory.length ? draft : commandHistory[historyIndex];
});
el('version').addEventListener('change', ()=>{
  for (const card of document.querySelectorAll('.version-card')) card.hidden = card.dataset.version !== el('version').value;
  print(`trid > 모델 버전: ${el('version').selectedOptions[0].textContent} · 다음 질문부터 적용됩니다.`, 'muted');
});
el('version-form').addEventListener('submit', event=>event.preventDefault());
el('file').addEventListener('change', ()=>{
  const file = el('file').files[0]; if (!file || busy) return;
  if (file.size > 2*1024*1024) {print('오류 > PDF는 2MB 이하로 넣어 주세요.', 'error'); return;}
  void task(async()=>loaded(await api('/api/load', file, {'Content-Type':'application/pdf','X-File-Name':encodeURIComponent(file.name)})));
});
el('save-text').onclick = ()=>void task(async()=>{
  loaded(await jsonApi('/api/text', {text:el('document-text').value}));
  el('document-text').value = ''; pastePanel(false);
});
el('cancel-text').onclick = ()=>{el('document-text').value = ''; pastePanel(false);};
el('document-text').addEventListener('keydown', event=>{
  if (busy || event.isComposing) return;
  if (event.key === 'Escape') {event.preventDefault(); el('cancel-text').click();}
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {event.preventDefault(); el('save-text').click();}
});
print('TRID > 안녕하세요. 무엇을 도와드릴까요? 업무 규정이나 기준을 물으시면 공개 자료에서 근거를 찾아 드립니다.', 'muted');
