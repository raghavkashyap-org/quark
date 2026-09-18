/**
 * Mock OpenAI-compatible server (Groq / OpenRouter / Ollama shaped).
 * Used by tests/e2e-providers.mjs to prove the provider adapter and the
 * automatic quota failover work without spending real credits.
 *
 *   POST /v1/chat/completions
 *   Turn 1 (no tool result yet) → asks for the get_datetime tool
 *   Turn 2                      → answers, echoing which provider served it
 */
import http from 'node:http';

const tools0 = (payload) => (Array.isArray(payload?.tools) ? payload.tools : []);

const PORT = Number(process.env.MOCK_OPENAI_PORT || 9998);
const MARKER = '[openai-provider]';

/** Every request the proxy actually put on the wire — inspected by the tests. */
const seen = [];

const server = http.createServer((req, res) => {
  if (req.method !== 'POST') {
    // GET /mock/seen → what was sent (tool counts, payload size, model).
    if (req.url?.startsWith('/mock/seen')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ ok: true, count: seen.length, requests: seen }));
    }
    // GET /v1/models → the catalogue a real host serves. Deliberately missing
    // the retired llama ids, exactly like Groq after 2026-08-16.
    if (req.url?.startsWith('/v1/models')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ object: 'list', data: [
        { id: 'openai/gpt-oss-120b', object: 'model' },
        { id: 'openai/gpt-oss-20b', object: 'model' },
        { id: 'qwen/qwen3.6-27b', object: 'model' },
        { id: 'meta-llama/llama-guard-4-12b', object: 'model' },
        { id: 'groq/compound', object: 'model' },
        { id: 'whisper-large-v3-turbo', object: 'model' },
      ] }));
    }
    if (req.url?.startsWith('/mock/reset')) {
      seen.length = 0;
      res.writeHead(200, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ ok: true }));
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(JSON.stringify({ ok: true, service: 'mock-openai', port: PORT }));
  }
  let body = '';
  req.on('data', (c) => { body += c; });
  req.on('end', () => {
    const auth = req.headers.authorization || '';
    if (!/^Bearer\s+\S+/.test(auth)) {
      res.writeHead(401, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: 'Missing bearer token' } }));
    }

    let payload = {};
    try { payload = JSON.parse(body || '{}'); } catch { /* ignore */ }
    const messages = payload.messages || [];

    // Sanity checks the real hosts also enforce.
    if (!Array.isArray(messages) || !messages.length) {
      res.writeHead(400, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: "messages is required" } }));
    }

    seen.push({
      at: Date.now(),
      model: payload.model || null,
      tools: tools0(payload).length,
      toolNames: tools0(payload).map((t) => t?.function?.name).filter(Boolean),
      systemChars: (messages.find((m) => m.role === 'system')?.content || '').length,
      bytes: Buffer.byteLength(body || ''),
      stream: payload.stream === true,
      temperature: payload.temperature ?? null,
      // Which output-limit field went on the wire — Groq rejects the wrong one.
      tokenParam: 'max_completion_tokens' in payload ? 'max_completion_tokens'
        : 'max_tokens' in payload ? 'max_tokens' : null,
      tokenValue: payload.max_completion_tokens ?? payload.max_tokens ?? null,
      hasParallelToolCalls: 'parallel_tool_calls' in payload,
      reasoningFormat: payload.reasoning_format ?? null,
      recovered: null,
    });

    // ── injected host-compatibility failures ────────────────────────────
    // Real hosts 400 on parameters they do not accept and on malformed tool
    // calls. These let the proxy's self-healing be tested without credits.
    const lastUserText = [...messages].reverse().find((m) => m.role === 'user')?.content || '';
    const bad = (message, code) => {
      res.writeHead(400, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message, type: 'invalid_request_error', ...(code ? { code } : {}) } }));
    };
    if (/retiredmodel/.test(String(lastUserText)) && /llama-3\.3-70b|llama-3\.1-8b/.test(String(payload.model || ''))) {
      res.writeHead(400, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: {
        message: `The model \`${payload.model}\` does not exist or you do not have access to it.`,
        type: 'invalid_request_error',
        code: 'model_not_found',
      } }));
    }
    if (/badparam/.test(String(lastUserText)) && 'parallel_tool_calls' in payload) {
      return bad("Unsupported parameter: 'parallel_tool_calls' is not supported with this model.");
    }
    if (/failtools/.test(String(lastUserText)) && (payload.temperature ?? 1) > 0.3 && tools0(payload).length) {
      return bad("Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.", 'tool_use_failed');
    }
    if (/notools/.test(String(lastUserText)) && tools0(payload).length) {
      return bad("Failed to call a function. Please adjust your prompt.", 'tool_use_failed');
    }
    seen[seen.length - 1].recovered = /failtools|notools|badparam/.test(String(lastUserText)) ? 'survived' : null;

    const hasToolResult = messages.some((m) => m.role === 'tool');
    const lastUser = [...messages].reverse().find((m) => m.role === 'user')?.content || '';
    const tools = Array.isArray(payload.tools) ? payload.tools : [];

    if (!hasToolResult && tools.length) {
      res.writeHead(200, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({
        id: 'chatcmpl-mock',
        object: 'chat.completion',
        model: payload.model || 'mock-model',
        choices: [{
          index: 0,
          finish_reason: 'tool_calls',
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [{
              id: 'call_mock_1',
              type: 'function',
              function: { name: 'get_datetime', arguments: JSON.stringify({ part: 'time' }) },
            }],
          },
        }],
        usage: { prompt_tokens: 500, completion_tokens: 18, total_tokens: 518 },
      }));
    }

    const toolText = messages.filter((m) => m.role === 'tool').map((m) => m.content).join(' ');
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({
      id: 'chatcmpl-mock',
      object: 'chat.completion',
      model: payload.model || 'mock-model',
      choices: [{
        index: 0,
        finish_reason: 'stop',
        message: {
          role: 'assistant',
          content: `${MARKER} Served by ${payload.model}. You asked: "${String(lastUser).slice(0, 60)}". Tool said: ${toolText.slice(0, 80)}`,
        },
      }],
      usage: { prompt_tokens: 620, completion_tokens: 40, total_tokens: 660 },
    }));
  });
});

server.listen(PORT, '127.0.0.1', () => console.log(`mock openai-compatible provider on ${PORT}`));
