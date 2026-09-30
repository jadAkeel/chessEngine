export const PROVIDERS = [
  {
    value: 'custom', label: 'Custom · OpenAI-compatible', protocol: 'openai_compatible', needsUrl: true,
    path: '/chat/completions', auth: 'Authorization: Bearer <key>', reasoningField: 'reasoning_effort',
    modelPlaceholder: 'Model ID from your provider',
    help: 'Any public HTTPS API that implements Chat Completions. The app sends POST {base URL}/chat/completions.',
    examples: [
      { url: 'https://api.moonshot.ai/v1', label: 'Kimi (Moonshot)' },
      { url: 'https://api.z.ai/api/paas/v4', label: 'Z.ai / GLM' },
      { url: 'https://api.deepseek.com', label: 'DeepSeek' },
      { url: 'https://api.mistral.ai/v1', label: 'Mistral' },
      { url: 'https://api.groq.com/openai/v1', label: 'Groq' },
      { url: 'https://api.together.xyz/v1', label: 'Together AI' },
      { url: 'https://api.x.ai/v1', label: 'xAI' },
    ],
  },
  {
    value: 'custom_anthropic', label: 'Custom · Anthropic-compatible', protocol: 'anthropic_compatible', needsUrl: true,
    path: '/messages', auth: 'x-api-key: <key>', reasoningField: 'thinking.budget_tokens (adaptive thinking + effort for Claude 4.6+ IDs)',
    modelPlaceholder: 'Model ID from your provider',
    help: 'Any public HTTPS API that implements the Anthropic Messages format. The app sends POST {base URL}/messages. If your provider documents an ANTHROPIC_BASE_URL for SDKs, add /v1 to it here.',
    examples: [
      { url: 'https://api.moonshot.ai/anthropic/v1', label: 'Kimi (Moonshot)' },
      { url: 'https://api.z.ai/api/anthropic/v1', label: 'Z.ai / GLM' },
      { url: 'https://api.deepseek.com/anthropic/v1', label: 'DeepSeek' },
    ],
  },
  {
    value: 'openai', label: 'OpenAI', protocol: 'openai_compatible', fixedUrl: 'https://api.openai.com/v1',
    path: '/chat/completions', auth: 'Authorization: Bearer <key>', reasoningField: 'reasoning_effort',
    modelPlaceholder: 'e.g. gpt-4.1-mini', help: 'OpenAI Chat Completions API.',
  },
  {
    value: 'openrouter', label: 'OpenRouter', protocol: 'openai_compatible', fixedUrl: 'https://openrouter.ai/api/v1',
    path: '/chat/completions', auth: 'Authorization: Bearer <key>', reasoningField: 'reasoning_effort',
    modelPlaceholder: 'e.g. openai/gpt-4.1-mini', help: 'OpenRouter Chat Completions API. Use the full vendor/model ID.',
  },
  {
    value: 'gemini', label: 'Google Gemini', protocol: 'gemini', fixedUrl: 'https://generativelanguage.googleapis.com/v1beta',
    path: '/models/{model}:generateContent', auth: 'x-goog-api-key: <key>', reasoningField: 'thinkingConfig (thinkingLevel for Gemini 3+, thinkingBudget for older)',
    modelPlaceholder: 'e.g. gemini-2.5-flash', help: 'Google Gemini generateContent API.',
  },
  {
    value: 'anthropic', label: 'Anthropic', protocol: 'anthropic', fixedUrl: 'https://api.anthropic.com/v1',
    path: '/messages', auth: 'x-api-key: <key>', reasoningField: 'adaptive thinking + output_config.effort (thinking budget for Claude 4.5 and older)',
    modelPlaceholder: 'e.g. claude-sonnet-5-5', help: 'Anthropic Messages API.',
  },
]

export const INITIAL_PLAYER = {
  provider: 'custom', model: '', api_key: '', base_url: '', token_parameter: 'auto',
  reasoning: 'default', temperature: '', max_tokens: 16000, timeout_seconds: 120,
}

export const SETTINGS_PRESETS = [
  { value: 'economy', label: 'Economy', max_tokens: 2048, timeout_seconds: 120, description: 'Smaller output cap; reasoning models may run out of tokens before returning a move.' },
  { value: 'balanced', label: 'Balanced · recommended starting point', max_tokens: 4096, timeout_seconds: 180, description: 'A practical starting budget. Increase it if the model runs out of output tokens.' },
  { value: 'extended', label: 'Extended thinking', max_tokens: 16000, timeout_seconds: 240, description: 'More room and time for reasoning; can cost more. Provider limits still apply.' },
]

export function applySettingsPreset(player, name) {
  const preset = SETTINGS_PRESETS.find(item => item.value === name)
  if (!preset) return player
  return { ...player, reasoning: 'default', temperature: '', max_tokens: preset.max_tokens, timeout_seconds: preset.timeout_seconds }
}

export function normalizedTemperature(player) {
  const value = player.temperature
  return value == null || (typeof value === 'string' && !value.trim()) ? null : Number(value)
}

export function selectedSettingsPreset(player) {
  return SETTINGS_PRESETS.find(item =>
    player.reasoning === 'default' && normalizedTemperature(player) === null &&
    Number(player.max_tokens) === item.max_tokens && Number(player.timeout_seconds) === item.timeout_seconds,
  )?.value || 'custom'
}

export const providerInfo = value => PROVIDERS.find(item => item.value === value) || PROVIDERS[0]

export function resolvedBaseUrl(player) {
  const info = providerInfo(player.provider)
  return info.fixedUrl || player.base_url.trim().replace(/\/+$/, '')
}

export function endpointPreview(player) {
  const info = providerInfo(player.provider)
  const base = resolvedBaseUrl(player)
  if (!base) return ''
  return base + info.path.replace('{model}', player.model.trim() || '{model}')
}

export function requestPlayer(player) {
  const info = providerInfo(player.provider)
  return {
    provider: info.protocol,
    model: player.model.trim(),
    api_key: player.api_key.trim(),
    base_url: info.needsUrl ? resolvedBaseUrl(player) : info.value === 'openrouter' ? info.fixedUrl : '',
    reasoning: player.reasoning,
    temperature: normalizedTemperature(player),
    token_parameter: info.protocol === 'openai_compatible' ? player.token_parameter : 'auto',
    max_tokens: Number(player.max_tokens),
    timeout_seconds: Number(player.timeout_seconds),
  }
}

// Key-free description for reports and PGN headers.
export function publicPlayer(player) {
  const info = providerInfo(player.provider)
  return {
    provider: info.label, protocol: info.protocol, base_url: resolvedBaseUrl(player) || null,
    model: player.model.trim(), reasoning: player.reasoning,
    temperature: normalizedTemperature(player),
    token_parameter: info.protocol === 'openai_compatible' ? player.token_parameter : 'auto',
    max_tokens: Number(player.max_tokens), timeout_seconds: Number(player.timeout_seconds),
  }
}

export function baseUrlProblem(raw) {
  const value = raw.trim()
  if (!value) return 'Enter the provider API base URL.'
  let url
  try { url = new URL(value) } catch { return 'Enter a full URL, e.g. https://api.example.com/v1.' }
  if (url.protocol !== 'https:') return 'Use an https:// URL.'
  if (url.username || url.password) return 'Remove credentials from the URL; paste the key in the API key field.'
  if (url.port && url.port !== '443') return 'Custom ports are not allowed; use the standard HTTPS port.'
  if (url.search || url.hash) return 'Remove query strings and fragments from the URL.'
  if (/\/(chat\/completions|messages)\/?$/.test(url.pathname)) return 'Enter the base URL only; the endpoint path is added automatically.'
  return ''
}

const COMPARED = [
  { field: 'reasoning', label: 'Reasoning effort', value: player => player.reasoning },
  { field: 'temperature', label: 'Temperature', value: normalizedTemperature },
  { field: 'max_tokens', label: 'Max output tokens', value: player => Number(player.max_tokens) },
  { field: 'timeout_seconds', label: 'Timeout', value: player => Number(player.timeout_seconds) },
]

// Budget settings that differ between the models, which makes a result less like-for-like.
export function settingDifferences(players) {
  return COMPARED
    .filter(item => item.value(players.A) !== item.value(players.B))
    .map(item => ({ field: item.field, label: item.label, A: item.value(players.A), B: item.value(players.B) }))
}

// Returns field-keyed problems; an empty object means the player can be used.
export function playerProblems(player) {
  const info = providerInfo(player.provider)
  const problems = {}
  const model = player.model.trim()
  if (!model) problems.model = 'Enter a model ID.'
  else if (!/^[A-Za-z0-9._:/@+-]{1,120}$/.test(model)) problems.model = 'Model IDs may use letters, digits and . _ : / @ + - only.'
  if (!player.api_key.trim()) problems.api_key = 'Paste an API key.'
  if (info.needsUrl) {
    const problem = baseUrlProblem(player.base_url)
    if (problem) problems.base_url = problem
  }
  const tokens = Number(player.max_tokens)
  if (!Number.isInteger(tokens) || tokens < 256 || tokens > 64000) problems.max_tokens = 'Use a whole number from 256 to 64000.'
  const timeout = Number(player.timeout_seconds)
  if (!Number.isInteger(timeout) || timeout < 10 || timeout > 600) problems.timeout_seconds = 'Use a whole number from 10 to 600.'
  const temperature = normalizedTemperature(player)
  const anthropic = ['anthropic', 'anthropic_compatible'].includes(info.protocol)
  if (temperature !== null) {
    if (!Number.isFinite(temperature) || temperature < 0 || temperature > (anthropic ? 1 : 2)) {
      problems.temperature = `Use a number from 0 to ${anthropic ? 1 : 2}, or leave blank for the provider default.`
    } else if (anthropic && player.reasoning !== 'default' && temperature !== 1) {
      problems.temperature = 'With Anthropic thinking, leave temperature blank or set it to 1.'
    }
  }
  return problems
}
