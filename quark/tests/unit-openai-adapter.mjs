/**
 * Unit tests — the OpenAI-compatible request adapter.
 *
 * These exist because of a real production failure: Groq removed the legacy
 * `max_tokens` parameter, so every request 400'd and Q.U.A.R.K. silently fell
 * back to the local core ("Groq is attached but nothing works"). The parameter
 * name is chosen per host, and this pins that decision down.
 *
 * Run: node tests/unit-openai-adapter.mjs
 */
import {
  toOpenAIRequest, toOpenAITools, toOpenAIMessages,
  isModelNotFound, pickReplacementModel, RETIRED_MODELS, MODEL_PREFERENCE, classifyOpenAI,
  capOutputTokens, TPM_SAFE_OUTPUT_TOKENS,
} from '../lib/llm-openai.js';

const results = [];
const check = (n, p, d = '') => { results.push({ n, p, d }); console.log(`${p ? '✓' : '✗'} ${n}${d ? ` — ${d}` : ''}`); };

const base = {
  contents: [{ role: 'user', parts: [{ text: 'what time is it' }] }],
  systemPrompt: 'You are Q.U.A.R.K.',
  declarations: [{ name: 'get_datetime', description: 'now', parameters: { type: 'OBJECT', properties: {} } }],
  maxTokens: 8192,
};

// ── the regression that broke the live deployment ───────────────────────────
const groq = toOpenAIRequest({ ...base, model: 'llama-3.3-70b-versatile', base: 'https://api.groq.com/openai/v1' });
check('Groq gets max_completion_tokens', groq.max_completion_tokens === 8192, `got ${groq.max_completion_tokens}`);
check('Groq never gets the retired max_tokens', groq.max_tokens === undefined, `got ${groq.max_tokens}`);

// ── gpt-oss needs the new name on ANY host (Groq, OpenRouter, local) ────────
const oss = toOpenAIRequest({ ...base, model: 'openai/gpt-oss-120b', base: 'http://127.0.0.1:11434/v1' });
check('gpt-oss gets max_completion_tokens on a non-Groq host', oss.max_completion_tokens === 8192 && oss.max_tokens === undefined);

// ── everything else keeps the legacy name (Ollama / llama.cpp / LM Studio) ──
const ollama = toOpenAIRequest({ ...base, model: 'llama3.2:3b', base: 'http://127.0.0.1:11434/v1' });
check('Ollama keeps max_tokens', ollama.max_tokens === 8192 && ollama.max_completion_tokens === undefined);
const openrouter = toOpenAIRequest({ ...base, model: 'meta-llama/llama-3.3-70b-instruct', base: 'https://openrouter.ai/api/v1' });
check('OpenRouter keeps max_tokens', openrouter.max_tokens === 8192 && openrouter.max_completion_tokens === undefined);

// ── a missing/odd base must not crash the adapter ───────────────────────────
const noBase = toOpenAIRequest({ ...base, model: 'llama-3.1-8b-instant' });
check('no base URL still produces a sendable body', noBase.model === 'llama-3.1-8b-instant' && noBase.max_tokens === 8192);

// ── the rest of the request shape ───────────────────────────────────────────
check('model and messages are carried through', groq.model === 'llama-3.3-70b-versatile' && Array.isArray(groq.messages) && groq.messages.length >= 2,
  `messages=${groq.messages?.length}`);
check('the system prompt is the first message', groq.messages[0].role === 'system' && /Q\.U\.A\.R\.K\./.test(groq.messages[0].content));
check('tools are attached with auto choice', Array.isArray(groq.tools) && groq.tools.length === 1 && groq.tool_choice === 'auto',
  `tools=${groq.tools?.length}`);
check('tools are converted to the OpenAI function shape',
  toOpenAITools(base.declarations)[0]?.type === 'function' && toOpenAITools(base.declarations)[0]?.function?.name === 'get_datetime');
check('an empty tool list is omitted, not sent as []',
  toOpenAIRequest({ ...base, declarations: [], model: 'x', base: 'https://api.groq.com/openai/v1' }).tools === undefined);
check('messages convert Gemini-shaped parts', toOpenAIMessages(base.contents, 'sys')[1].role === 'user'
  && /what time is it/.test(toOpenAIMessages(base.contents, 'sys')[1].content));

// ── gpt-oss parameter hygiene (Groq rejects the wrong ones) ────────────────
const ossTools = toOpenAIRequest({ ...base, model: 'openai/gpt-oss-120b', base: 'https://api.groq.com/openai/v1' });
check('gpt-oss is not sent parallel_tool_calls (unsupported → 400)',
  ossTools.tools?.length === 1 && ossTools.parallel_tool_calls === undefined);
check('gpt-oss with tools sets reasoning_format=hidden (Groq requires parsed/hidden)',
  ossTools.reasoning_format === 'hidden', `got ${ossTools.reasoning_format}`);
const ossNoTools = toOpenAIRequest({ ...base, declarations: [], model: 'openai/gpt-oss-120b', base: 'https://api.groq.com/openai/v1' });
check('gpt-oss without tools leaves reasoning_format alone', ossNoTools.reasoning_format === undefined);
check('llama keeps parallel_tool_calls=false', groq.parallel_tool_calls === false && groq.reasoning_format === undefined);

// ── recognising a provider shutdown ─────────────────────────────────────────
check('Groq\'s 400 model_not_found is recognised',
  isModelNotFound(400, { error: { code: 'model_not_found', message: 'The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.' } }));
check('a plain 404 is recognised', isModelNotFound(404, null));
check('a 401 bad key is NOT mistaken for a retirement',
  !isModelNotFound(401, { error: { message: 'Invalid API key' } }));
check('a 400 malformed-tool-call is NOT mistaken for a retirement',
  !isModelNotFound(400, { error: { code: 'tool_use_failed', message: 'Failed to call a function.' } }));
check('the retirement table names Groq\'s own replacements',
  RETIRED_MODELS['llama-3.3-70b-versatile']?.use === 'openai/gpt-oss-120b'
    && RETIRED_MODELS['llama-3.1-8b-instant']?.use === 'openai/gpt-oss-20b');
check('the hint tells the operator exactly what to set',
  /OPENAI_MODEL=openai\/gpt-oss-120b/.test(classifyOpenAI(400, { error: { code: 'model_not_found' } }, 'llama-3.3-70b-versatile').hint),
  classifyOpenAI(400, { error: { code: 'model_not_found' } }, 'llama-3.3-70b-versatile').hint.slice(0, 90));

// ── picking a replacement from what a host actually serves ──────────────────
const catalogue = [{ id: 'openai/gpt-oss-20b' }, { id: 'qwen/qwen3.6-27b' }, { id: 'groq/compound' },
  { id: 'meta-llama/llama-guard-4-12b' }, { id: 'whisper-large-v3-turbo' }];
check('prefers the best tool-capable model the host offers',
  pickReplacementModel(catalogue, 'llama-3.3-70b-versatile') === 'qwen/qwen3.6-27b',
  `got ${pickReplacementModel(catalogue, 'llama-3.3-70b-versatile')}`);
check('never picks a model with no function calling',
  !/compound|guard|whisper/.test(String(pickReplacementModel(catalogue, 'llama-3.1-8b-instant'))));
check('accepts a plain string catalogue too',
  pickReplacementModel(['openai/gpt-oss-120b', 'groq/compound'], 'llama-3.3-70b-versatile') === 'openai/gpt-oss-120b');
check('will not "replace" a model with itself',
  pickReplacementModel(['openai/gpt-oss-120b'], 'openai/gpt-oss-120b') === null);
check('with no catalogue it falls back to the documented replacement',
  pickReplacementModel(null, 'llama-3.3-70b-versatile') === 'openai/gpt-oss-120b');
check('an unknown model with no catalogue yields no guess', pickReplacementModel([], 'some/custom-model') === null);
check('every preferred model is tool-capable by Groq\'s own table',
  MODEL_PREFERENCE.every((m) => !/compound|guard|safeguard|whisper|allam|orpheus/.test(m)), MODEL_PREFERENCE.join(','));

// ── tokens-per-minute guard ────────────────────────────────────────────────
const cap = capOutputTokens({ base: 'https://api.groq.com/openai/v1', requested: 8192 });
check('an output ask bigger than Groq\'s 8K TPM is clamped',
  cap.value === TPM_SAFE_OUTPUT_TOKENS && cap.capped === true, `value=${cap.value} capped=${cap.capped}`);
check('a sensible ask is left alone',
  capOutputTokens({ base: 'https://api.groq.com/openai/v1', requested: 1024 }).value === 1024
    && capOutputTokens({ base: 'https://api.groq.com/openai/v1', requested: 1024 }).capped === false);
check('non-Groq hosts are not clamped',
  capOutputTokens({ base: 'http://127.0.0.1:11434/v1', requested: 8192 }).value === 8192);
check('QUARK_IGNORE_TPM_CAP=1 opts out',
  capOutputTokens({ base: 'https://api.groq.com/openai/v1', requested: 8192, ignoreCap: true }).value === 8192);
check('a missing or invalid value falls back to the safe default',
  capOutputTokens({ base: 'https://api.groq.com/openai/v1', requested: NaN }).value === TPM_SAFE_OUTPUT_TOKENS);
check('the clamp reaches the request body',
  toOpenAIRequest({ ...base, model: 'openai/gpt-oss-120b', base: 'https://api.groq.com/openai/v1',
    maxTokens: capOutputTokens({ base: 'https://api.groq.com/openai/v1', requested: 8192 }).value }).max_completion_tokens === TPM_SAFE_OUTPUT_TOKENS);

// ── summary ─────────────────────────────────────────────────────────────────
const passed = results.filter((r) => r.p).length;
console.log(`\n${'═'.repeat(34)}\n  ${passed}/${results.length} adapter checks passed\n${'═'.repeat(34)}`);
if (passed !== results.length) {
  console.log('\nFailed:');
  results.filter((r) => !r.p).forEach((r) => console.log(` • ${r.n} — ${r.d}`));
}
process.exit(passed === results.length ? 0 : 1);
