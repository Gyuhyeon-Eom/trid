'use strict';
const el = id => document.getElementById(id);
const token = document.querySelector('meta[name="trid-token"]').content;
let documentId = null, pages = [], documentName = '', busy = false;
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
  print(`trid > ${documentName} · ${pages.length}페이지를 읽었습니다.\n질문을 입력하세요. /source로 원문을 확인할 수 있습니다.`, 'reply');
  for (const warning of data.warnings) print(warning.replace('추출 문서 보기에서 확인해 주세요.', '/source로 원문을 확인해 주세요.'), 'muted');
}
function help() {
  print('/sample  가상 출장 규정 불러오기\n/pdf     텍스트 PDF 열기 (2MB · 20페이지 · 5만 자 이하, OCR 미지원)\n/paste   문서 직접 붙여넣기 (5,000자 이하)\n/source  현재 문서 원문 보기\n/clear   서버 문서와 이 화면의 기록 지우기\n/help    도움말\n\n문서를 넣은 뒤 질문을 그대로 입력하세요.\n예: 해외 출장 식비는 얼마야?\n이전 질문을 이어서 이해하는 대화 모드가 아니라, 질문마다 문서에서 답을 찾습니다.\n↑ / ↓ 이전 명령 선택 · Enter 실행', 'muted');
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
    print(documentId ? `${documentName}\n\n${pages.map((p,i)=>`[${i+1}페이지]\n${p}`).join('\n\n')}` : 'trid > 먼저 /sample, /pdf 또는 /paste로 문서를 넣어 주세요.', 'muted');
    return;
  }
  if (value === '/sample') {await task(async()=>loaded(await jsonApi('/api/sample', {}))); return;}
  if (value === '/clear' || value === '/reset') {
    await task(async()=>{
      await jsonApi('/api/clear', {});
      documentId = null; pages = []; documentName = ''; commandHistory = []; historyIndex = 0; draft = '';
      el('file').value = ''; el('document-text').value = ''; el('output').replaceChildren();
      print('trid > 문서와 화면 기록을 지웠습니다. /sample, /pdf 또는 /paste로 다시 시작하세요.');
    }); return;
  }
  if (value.startsWith('/')) {print('trid > 알 수 없는 명령입니다. /help로 명령어를 확인하세요.', 'muted'); return;}
  if (!documentId) {print('trid > 먼저 /sample, /pdf 또는 /paste로 문서를 넣어 주세요.'); return;}
  await task(async()=>{
    const started = performance.now();
    const result = await jsonApi('/api/ask', {question:value, document_id:documentId});
    print(`trid > ${result.answer}`, 'reply');
    print(`[서버 처리 ${result.elapsed_ms}ms · 왕복 ${Math.round(performance.now()-started)}ms · ${result.checked_chunks}/${result.total_chunks}개 구간 확인]`, 'muted');
    for (const candidate of result.candidates) {
      const [a,b] = candidate.prediction.source_character_span;
      const chars = Array.from(candidate.text);
      // Code-point offsets are returned by the Python model; bracket the exact source span.
      const source = chars.slice(0,a).join('') + '【' + chars.slice(a,b).join('') + '】' + chars.slice(b).join('');
      print(`[근거 ${candidate.page}페이지]\n${source}`, 'muted');
    }
    if (!result.candidates.length) print(result.scope, 'muted');
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
print('trid > /sample을 입력하면 바로 테스트할 수 있습니다.');
