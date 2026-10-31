"""Model-free stand-in for llama-server, for the AWS-EVAL-006 public fixture check ONLY (scripts/aws_eval/fixture_e2e.sh).

It answers the five endpoints base_dialogue_runner uses (/props, /apply-template, /tokenize, /completion, /health) with
the package's own official tokenizer and chat template, and returns a deterministic placeholder reply per prompt. No
model is loaded and nothing is generated: render parity is trivially true here and the replies mean nothing. It exists
so the delivery → runner → encrypted checkpoint → collector path can be exercised on any CPU without a GPU or a model.

    fake_llama_server.sh -m runs/chat-blind-005/packages/P/model.gguf --port 51990 ...   (other llama-server flags ignored)
"""
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('-c', type=int, default=4096)
    a, _ = p.parse_known_args()
    tok = AutoTokenizer.from_pretrained(Path(a.m).parent / 'tokenizer')  # the package's official tokenizer alias

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/props':
                self.send({'model_path': a.m, 'chat_template': tok.chat_template or '', 'bos_token': tok.bos_token or '',
                           'eos_token': tok.eos_token or '', 'build_info': 'fixture-fake-llama-server (no model)',
                           'default_generation_settings': {'params': {}, 'n_ctx': a.c}})
            else:
                self.send({'status': 'ok'})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/apply-template':
                self.send({'prompt': tok.apply_chat_template(req['messages'], tokenize=False, add_generation_prompt=True,
                                                             **req.get('chat_template_kwargs', {}))})
            elif self.path == '/tokenize':
                self.send({'tokens': tok(req['content'], add_special_tokens=req.get('add_special', True))['input_ids']})
            elif self.path == '/completion':
                digest = hashlib.sha256(json.dumps(req['prompt']).encode()).hexdigest()[:12]
                text = f'픽스처 응답 {digest} (모델 없음)'
                settings = {k: v for k, v in req.items() if k not in ('prompt', 'return_tokens', 'cache_prompt')}
                self.send({'content': text, 'tokens': tok(text, add_special_tokens=False)['input_ids'], 'stop_type': 'eos',
                           'stopping_word': '', 'generation_settings': {**settings, 'fixture': True},
                           'timings': {'predicted_per_second': 0.0, 'prompt_n': len(req['prompt'])}})
            else:
                self.send_error(404)

    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
