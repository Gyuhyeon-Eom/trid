const HOST = 'test.tridatum.co';
const GET_PATHS = new Set(['/', '/app.js', '/healthz']);
const POST_PATHS = new Set(['/api/sample', '/api/text', '/api/load', '/api/ask', '/api/clear', '/api/pin']);

function error(status, message) {
  return Response.json({error: message}, {status, headers: {'Cache-Control': 'no-store'}});
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.hostname !== HOST) return error(404, '없는 사이트입니다.');
    if (url.protocol !== 'https:') return Response.redirect(`https://${HOST}${url.pathname}`, 308);
    if (!(request.method === 'GET' && GET_PATHS.has(url.pathname)) &&
        !(request.method === 'POST' && POST_PATHS.has(url.pathname))) return error(404, '없는 기능입니다.');
    if (!env.ORIGIN_URL || !env.PROXY_SECRET) return error(503, '테스트 서버를 준비 중입니다.');
    if (request.method === 'POST') {
      if (request.headers.get('Origin') !== `https://${HOST}`) return error(403, '허용되지 않은 요청입니다.');
      const length = Number(request.headers.get('Content-Length'));
      if (!request.headers.has('Content-Length') || !Number.isInteger(length) || length < 0 || length > 2 * 1024 * 1024)
        return error(413, '입력은 2MB 이하로 넣어 주세요.');
    }
    if (env.DEMO_RATE_LIMIT) {
      const {success} = await env.DEMO_RATE_LIMIT.limit({key: request.headers.get('CF-Connecting-IP') || 'unknown'});
      if (!success) return error(429, '요청이 많습니다. 1분 뒤 다시 시도해 주세요.');
    }
    // Forward only the explicitly needed browser headers, then add trusted proxy identity.
    const headers = new Headers();
    for (const key of ['Content-Type', 'Content-Length', 'Cookie', 'Origin', 'X-Trid-Token', 'X-File-Name']) {
      if (request.headers.has(key)) headers.set(key, request.headers.get(key));
    }
    headers.set('X-Trid-Proxy', env.PROXY_SECRET);
    headers.set('X-Trid-Host', HOST);
    try {
      const body = request.method === 'POST' ? await request.arrayBuffer() : undefined;
      if (body && body.byteLength > 2 * 1024 * 1024) return error(413, '입력은 2MB 이하로 넣어 주세요.');
      const upstream = await fetch(new URL(url.pathname, env.ORIGIN_URL), {
        method: request.method, headers, body,
        // CPU generation over retrieved passages can take tens of seconds.
        redirect: 'manual', signal: AbortSignal.timeout(120000),
      });
      if (upstream.status >= 300 && upstream.status < 400) throw new Error('Origin redirect refused');
      if (upstream.status >= 500 && !upstream.headers.get('content-type')?.includes('application/json'))
        throw new Error(`Origin unavailable (${upstream.status})`);
      const response = new Response(upstream.body, upstream);
      response.headers.set('Cache-Control', 'no-store');
      response.headers.set('Strict-Transport-Security', 'max-age=86400');
      return response;
    } catch (failure) {
      console.error('Demo origin failure:', failure.name, failure.message);
      if (request.method === 'GET' && url.pathname === '/') return new Response(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>trid · 잠시 쉬는 중</title><body style="font-family:system-ui;padding:10vh 8vw;background:#f4f7f5;color:#192d29"><h1>trid 테스트 서버가 잠시 쉬고 있습니다.</h1><p>운영자의 Mac에서 실행하는 임시 데모입니다. 잠시 후 다시 접속해 주세요.</p></body></html>',
        {status: 503, headers: {'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store'}});
      return error(503, '테스트 서버가 꺼져 있거나 연결이 끊겼습니다. 잠시 후 다시 시도해 주세요.');
    }
  },
};
