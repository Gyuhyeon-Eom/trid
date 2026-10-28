import test from 'node:test';
import assert from 'node:assert/strict';
import worker from './worker.mjs';

const env = {ORIGIN_URL: 'https://origin.example', PROXY_SECRET: 'proxy-secret'};

test('rejects foreign origins and paths before contacting the model', async () => {
  const old = globalThis.fetch;
  globalThis.fetch = () => { throw new Error('Must not reach origin'); };
  try {
    assert.equal((await worker.fetch(new Request('https://test.tridatum.co/runs/model.safetensors'), env)).status, 404);
    assert.equal((await worker.fetch(new Request('https://test.tridatum.co/api/ask', {
      method: 'POST', headers: {Origin: 'https://evil.example', 'Content-Length': '2'}, body: '{}',
    }), env)).status, 403);
  } finally { globalThis.fetch = old; }
});

test('proxy replaces spoofed identity, strips unrelated headers and never follows redirects', async () => {
  const old = globalThis.fetch;
  let captured;
  globalThis.fetch = async (url, options) => {
    captured = {url, options};
    return new Response('{"ok":true}', {headers: {'Content-Type': 'application/json'}});
  };
  try {
    const result = await worker.fetch(new Request('https://test.tridatum.co/api/ask?ignored=yes', {
      method: 'POST', body: '{}', headers: {Origin: 'https://test.tridatum.co', 'Content-Length': '2',
        'X-Trid-Proxy': 'attacker', 'X-Trid-Host': 'evil.example', Authorization: 'irrelevant'},
    }), env);
    assert.equal(result.status, 200);
    assert.equal(captured.url.href, 'https://origin.example/api/ask');
    assert.equal(captured.options.headers.get('X-Trid-Proxy'), env.PROXY_SECRET);
    assert.equal(captured.options.headers.get('X-Trid-Host'), 'test.tridatum.co');
    assert.equal(captured.options.headers.get('Authorization'), null);
    assert.equal(captured.options.redirect, 'manual');
    assert.equal(captured.options.body.byteLength, 2);
    assert.equal(result.headers.get('Cache-Control'), 'no-store');
  } finally { globalThis.fetch = old; }
});

test('disconnected origin presents an honest offline page', async () => {
  const old = globalThis.fetch, oldLog = console.error;
  globalThis.fetch = async () => { throw new Error('offline'); };
  console.error = () => {};
  try {
    const response = await worker.fetch(new Request('https://test.tridatum.co/'), env);
    assert.equal(response.status, 503);
    assert.match(await response.text(), /임시 데모/);
  } finally { globalThis.fetch = old; console.error = oldLog; }
});
