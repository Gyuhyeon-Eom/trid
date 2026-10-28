"""Recognize terminal controls before sending ordinary text to the model."""
import re


def classify_input(text):
    text = text.strip()
    if not text:
        return "empty", ""
    if "\x1b" in text or text.startswith("^[["):
        return "notice", "방향키 제어 문자는 질문으로 처리하지 않습니다. 질문을 다시 입력하세요."
    command = re.fullmatch(r"/\s*(exit|reset|example|help)", text)
    if command:
        return "command", command.group(1)
    if re.match(r"^(?:\./)?\.venv/bin/|^(?:python(?:3)?|uv|pip(?:3)?|cd|source|bash|zsh)\s", text):
        return "notice", "터미널 명령으로 보입니다. /exit로 채팅을 종료한 뒤 % 또는 $ 프롬프트에서 실행하세요. 이 입력은 실행하거나 모델에 보내지 않았습니다."
    if text.startswith("/"):
        return "notice", "알 수 없는 명령입니다. /exit 종료 · /reset 초기화 · /example 예제 · /help 도움말"
    return "question", text
