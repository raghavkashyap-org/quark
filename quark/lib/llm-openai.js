/**
 * Q.U.A.R.K. — OpenAI-compatible provider adapter
 * ─────────────────────────────────────────────────────────────────────────
 * One adapter covers every "OpenAI-shaped" host, which is where the free and
 * self-hosted models live:
 *
 *   Groq          https://api.groq.com/openai/v1        (free tier, very fast)
 *   OpenRouter    https://openrouter.ai/api/v1          (free model variants)
 *   Together      https://api.together.xyz/v1
 *   Ollama        http://127.0.0.1:11434/v1             (your own machine)
 *   llama.cpp     http://127.0.0.1:8080/v1
 *   LM Studio     http://127.0.0.1:1234/v1
 *
 * It converts our internal Gemini-shaped `contents` into OpenAI `messages` and
 * maps the reply back into the shape `lib/gemini-client.js` already expects, so
 * the browser client needs zero changes when the provider switches.
 *
 * This module never reads process.env — the route passes config in.
 */

/** Gemini JSON-Schema declarations → OpenAI `tools`. */
export function toOpenAITools(declarations) {
  return (declarations || [])
    .filter((d) => d && d.name)
    .map((d) => ({
      type: 'function',
      function: {
        name: d.name,
        description: d.description || '',
        parameters: d.parameters || { type: 'object', properties: {} },
      },
    }));
}

function safeParseArgs(raw) {
  if (raw == null) return {};
  if (typeof raw === 'object') return raw;
  try {
    const parsed = JSON.parse(String(raw));
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

/**
 * Gemini `contents` → OpenAI `messages`.
 *
 * The awkward part is tool-call identity: OpenAI requires every `tool` message
 * to reference the `tool_call_id` of the assistant call it answers. Gemini only
 * sometimes supplies ids, so when they are missing we mint deterministic ones
 * and pair calls with responses positionally (they always arrive in order).
 */
export function toOpenAIMessages(contents, systemPrompt) {
  const messages = systemPrompt ? [{ role: 'system', content: systemPrompt }] : [];
  const pendingIds = [];

  for (const c of contents || []) {
    const parts = Array.isArray(c?.parts) ? c.parts : [];
    const text = parts.filter((p) => typeof p?.text === 'string').map((p) => p.text).join('\n').trim();
    const calls = parts.filter((p) => p?.functionCall);
    const responses = parts.filter((p) => p?.functionResponse);

    if (c.role === 'model') {
      if (calls.length) {
        const tool_calls = calls.map((p, i) => {
          const id = p.functionCall.id || `qk_${messages.length}_${i}`;
          pendingIds[i] = id;
          return {
            id,
            type: 'function',
            function: {
              name: p.functionCall.name,
              arguments: JSON.stringify(p.functionCall.args || {}),
            },
          };
        });
        messages.push({ role: 'assistant', content: text || null, tool_calls });
      } else if (text) {
        messages.push({ role: 'assistant', content: text });
      }
      continue;
    }

    // user turn — may be plain text, tool results, or both
    responses.forEach((p, i) => {
      messages.push({
        role: 'tool',
        tool_call_id: p.functionResponse.id || pendingIds[i] || `qk_${messages.length}_${i}`,
        content: JSON.stringify(p.functionResponse.response ?? {}),
      });
    });
    if (text) messages.push({ role: 'user', content: text });
  }

  return messages;
}

/** Build the chat-completions request body. */
export function toOpenAIRequest({ contents, systemPrompt, declarations, model, maxTokens, base }) {
  const tools = toOpenAITools(declarations);
  const body = {
    model,
    messages: toOpenAIMessages(contents, systemPrompt),
    temperature: 0.4,
  };
  const isGptOss = /gpt-oss/i.test(String(model || ''));
  if (tools.length) {
    body.tools = tools;
    body.tool_choice = 'auto';
    // gpt-oss does not support parallel tool calls and rejects the parameter
    // outright; every other host accepts `false`, which is what we want
    // (one tool call per round keeps the loop deterministic).
    if (!isGptOss) body.parallel_tool_calls = false;
    // Groq requires reasoning_format to be `parsed` or `hidden` when tools are
    // declared — with the default (`raw`) the model's chain of thought is
    // interleaved into `content` and tool calls come back malformed (400
    // `tool_use_failed`). `hidden` keeps the answer clean and the tokens low.
    if (isGptOss) body.reasoning_format = 'hidden';
  }
  if (maxTokens) {
    // Groq takes `max_completion_tokens` (it has removed the legacy
    // `max_tokens`); gpt-oss everywhere does too. Other OpenAI-compatible hosts
    // (Ollama, llama.cpp, LM Studio, OpenRouter) accept `max_tokens`. Sending
    // the wrong name is a hard 400 on every single request, so pick by host.
    const host = String(base || '');
    if (/gpt-oss/i.test(String(model || '')) || /groq/i.test(host)) body.max_completion_tokens = maxTokens;
    else body.max_tokens = maxTokens;
  }
  return body;
}

const FINISH_MAP = {
  stop: 'STOP',
  length: 'MAX_TOKENS',
  tool_calls: 'STOP',
  function_call: 'STOP',
  content_filter: 'SAFETY',
};

/** OpenAI response → the shape our client already understands. */
export function mapOpenAIResponse(json) {
  const choice = json?.choices?.[0];
  const msg = choice?.message || {};
  const raw = msg.content;
  const text =
    typeof raw === 'string'
      ? raw
      : Array.isArray(raw)
        ? raw.map((p) => (typeof p?.text === 'string' ? p.text : '')).join('')
        : '';

  const functionCalls = (msg.tool_calls || [])
    .filter((tc) => tc?.function?.name)
    .map((tc) => ({
      id: tc.id || null,
      name: tc.function.name,
      args: safeParseArgs(tc.function.arguments),
    }));

  const finishReason = FINISH_MAP[choice?.finish_reason] || choice?.finish_reason || null;
  const u = json?.usage;

  return {
    text: text.trim(),
    functionCalls,
    finishReason,
    blocked: finishReason === 'SAFETY',
    usage: u
      ? {
          promptTokens: u.prompt_tokens ?? 0,
          completionTokens: u.completion_tokens ?? 0,
          totalTokens: u.total_tokens ?? 0,
        }
      : null,
  };
}

/** Split text into a few pieces so the HUD still types the answer out. */
export function chunkText(text, pieces = 14) {
  const t = String(text || '');
  if (!t) return [];
  if (t.length <= 60 || pieces <= 1) return [t];
  const size = Math.ceil(t.length / pieces);
  const out = [];
  for (let i = 0; i < t.length; i += size) out.push(t.slice(i, i + size));
  return out;
}

/** Turn an upstream HTTP failure into something the HUD can explain. */
/**
 * Tokens-per-minute is the free-tier limit that actually bites, and the
 * *requested* output size counts against it whether or not the model uses it.
 * Groq's gpt-oss and qwen models allow 8,000 TPM: a 2,048-token output request
 * plus a ~1,600-token tool-carrying prompt is already ~3,600, so two rounds fit
 * in a minute. Asking for 8,192 would 429 on the very first request — which
 * looks exactly like "the key is set but only LOCAL CORE answers".
 */
export const TPM_SAFE_OUTPUT_TOKENS = 2048;

/** Clamp the output budget to what the host's free tier can serve per minute. */
export function capOutputTokens({ base, requested, ignoreCap = false }) {
  const asked = Number.isFinite(requested) && requested > 0 ? requested : TPM_SAFE_OUTPUT_TOKENS;
  if (ignoreCap || !/groq/i.test(String(base || ''))) return { value: asked, capped: false };
  return {
    value: Math.min(asked, TPM_SAFE_OUTPUT_TOKENS),
    capped: asked > TPM_SAFE_OUTPUT_TOKENS,
  };
}

/**
 * Models Groq has switched off, with the replacement Groq itself recommends.
 * Groq retires ids without a client-side warning: the request simply comes
 * back `model_not_found`, which reads like a typo rather than a shutdown.
 * Checked against console.groq.com/docs/deprecations on 2026-09-17.
 */
export const RETIRED_MODELS = {
  'llama-3.3-70b-versatile': { off: '2026-08-16', use: 'openai/gpt-oss-120b' },
  'llama-3.1-8b-instant': { off: '2026-08-16', use: 'openai/gpt-oss-20b' },
  'meta-llama/llama-4-scout-17b-16e-instruct': { off: '2026-07-17', use: 'openai/gpt-oss-120b' },
  'qwen/qwen3-32b': { off: '2026-07-17', use: 'openai/gpt-oss-120b' },
  'meta-llama/llama-4-maverick-17b-128e-instruct': { off: '2026-03-09', use: 'openai/gpt-oss-120b' },
  'gemma2-9b-it': { off: '2025-10-02', use: 'openai/gpt-oss-20b' },
};

/**
 * Preference order for automatic replacement: tool-capable, currently listed on
 * Groq's free/developer tier, best quality first. `groq/compound*` is excluded
 * on purpose — it does not support function calling at all.
 */
export const MODEL_PREFERENCE = [
  'openai/gpt-oss-120b',
  'qwen/qwen3.6-27b',
  'openai/gpt-oss-20b',
  'qwen/qwen3.8-27b',
  'minimaxai/minimax-m2.7',
  'moonshotai/kimi-k2-instruct-0905',
];

/** Is this a "that model id does not exist here" error? (404, or Groq's 400.) */
export function isModelNotFound(status, json) {
  const code = String(json?.error?.code || '');
  const msg = String(json?.error?.message || '');
  if (status === 404) return true;
  return /model_not_found/i.test(code)
    || /does not exist or you do not have access/i.test(msg)
    || /model .* (?:has been )?(?:decommissioned|deprecat|retired)/i.test(msg);
}

/**
 * Every plausible replacement, best first, from the ids a host actually serves
 * (`GET /models`). Several can be needed because "model_not_found" also means
 * *your key has no access* — a free account can be blocked from one id and
 * allowed the next, so trying a single replacement is not enough.
 */
export function replacementCandidates(available, requested, exclude = []) {
  const ids = (Array.isArray(available) ? available : [])
    .map((m) => (typeof m === 'string' ? m : m?.id))
    .filter(Boolean);
  const blocked = new Set([requested, ...exclude]);
  if (!ids.length) {
    const fallback = RETIRED_MODELS[requested]?.use;
    return fallback && !blocked.has(fallback) ? [fallback] : [];
  }
  const toolCapable = (id) => /gpt-oss|qwen|llama|kimi|minimax/i.test(id)
    && !/guard|safeguard|compound|whisper|embed|tts|playai|allam|orpheus|distill/i.test(id);
  const preferred = MODEL_PREFERENCE.filter((id) => ids.includes(id) && !blocked.has(id));
  const others = ids.filter((id) => !blocked.has(id) && toolCapable(id) && !preferred.includes(id));
  return [...preferred, ...others];
}

/** Best single replacement, or null when the host offers nothing usable. */
export function pickReplacementModel(available, requested, exclude = []) {
  return replacementCandidates(available, requested, exclude)[0] || null;
}

export function classifyOpenAI(status, json, model) {
  const msg = json?.error?.message || json?.message || '';
  if (status === 401 || status === 403) {
    return { code: 'unauthorized', hint: 'The OpenAI-compatible provider rejected the key (OPENAI_API_KEY).' };
  }
  if (isModelNotFound(status, json)) {
    const retired = RETIRED_MODELS[model];
    return {
      code: 'bad_model',
      hint: retired
        ? `Model "${model}" was switched off by the provider on ${retired.off}. Set OPENAI_MODEL=${retired.use} (Groq's recommended replacement) and redeploy.`
        : `Model "${model}" was not found at OPENAI_BASE_URL. Set OPENAI_MODEL to one the host lists at GET /models.`,
    };
  }
  if (status === 429) {
    return { code: 'rate_limited', hint: msg || 'That provider is rate-limiting this key.' };
  }
  if (status === 400) {
    return { code: 'bad_request', hint: msg || 'The provider rejected the request payload.' };
  }
  if (status >= 500) return { code: 'upstream', hint: `Provider error (HTTP ${status}).` };
  return { code: 'unknown', hint: msg || `HTTP ${status}` };
}
