import React, { useId, useState } from 'react'
import { CheckCircle2, ChevronDown, Eye, EyeOff, Loader2, PlugZap, XCircle } from 'lucide-react'
import {
  PROVIDERS, SETTINGS_PRESETS, applySettingsPreset, endpointPreview, normalizedTemperature,
  playerProblems, providerInfo, selectedSettingsPreset,
} from '../lib/providers.js'

const REASONING = [
  { value: 'default', label: 'Provider default' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]

function Field({ id, label, hint, error, full, children }) {
  return (
    <div className={`field ${full ? 'full' : ''}`}>
      <label htmlFor={id}>{label}</label>
      {children}
      {error ? <small id={`${id}-error`} className="field-error">{error}</small> : hint ? <small id={`${id}-hint`}>{hint}</small> : null}
    </div>
  )
}

export function ConnectionBadge({ verification }) {
  if (verification?.state === 'testing') return <span className="badge badge-busy"><Loader2 size={12} className="spin" aria-hidden="true" /> Testing</span>
  if (verification?.state === 'ok') return <span className="badge badge-ok"><CheckCircle2 size={12} aria-hidden="true" /> Verified</span>
  if (verification?.state === 'error') return <span className="badge badge-error"><XCircle size={12} aria-hidden="true" /> Test failed</span>
  return <span className="badge">Not tested</span>
}

export default function PlayerSetup({ model, colorNote, player, onChange, onVerify, verification, locked, keyEditable, showErrors }) {
  const uid = useId()
  const id = name => `${uid}-${name}`
  const [showKey, setShowKey] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const info = providerInfo(player.provider)
  const problems = playerProblems(player)
  const [touched, setTouched] = useState({})
  const errorFor = field => (showErrors || touched[field]) ? problems[field] : ''
  const update = (field, value) => onChange({ ...player, [field]: value })
  const touch = field => setTouched(previous => (previous[field] ? previous : { ...previous, [field]: true }))
  const describedBy = field => (errorFor(field) ? `${id(field)}-error` : `${id(field)}-hint`)
  const open = !locked || expanded
  const preview = endpointPreview(player)
  const testing = verification?.state === 'testing'
  const presetName = selectedSettingsPreset(player)
  const preset = SETTINGS_PRESETS.find(item => item.value === presetName)
  const anthropic = ['anthropic', 'anthropic_compatible'].includes(info.protocol)
  const gemini3 = info.protocol === 'gemini' && /^gemini-3/i.test(player.model.trim())
  const temperatureHint = gemini3
    ? <>Google recommends the provider default (1.0) for Gemini 3. <a href="https://ai.google.dev/gemini-api/docs/gemini-3?hl=en" target="_blank" rel="noreferrer">Provider guidance</a>.</>
    : anthropic
      ? 'Leave blank for widest compatibility. With thinking enabled, only blank or 1 is supported; newer models may only accept the default.'
      : 'Blank uses the provider default. Lower values usually vary less; some reasoning models reject custom temperature.'

  return (
    <section className={`setup-card model-${model.toLowerCase()}`} aria-labelledby={id('title')}>
      <div className="setup-heading">
        <span className={`model-chip chip-${model.toLowerCase()}`} aria-hidden="true">{model}</span>
        <div className="setup-title">
          <h3 id={id('title')}>Model {model}{player.model.trim() ? <span className="setup-model-name"> · {player.model.trim()}</span> : null}</h3>
          <span className="muted">{colorNote}</span>
        </div>
        <ConnectionBadge verification={verification} />
      </div>

      {locked && (
        <div className="locked-row">
          <span>{info.label} · reasoning {player.reasoning} · temperature {normalizedTemperature(player) ?? 'default'} · {player.max_tokens} tokens · {player.timeout_seconds}s timeout</span>
          <button type="button" className="text-button" aria-expanded={expanded} aria-controls={id('fields')} onClick={() => setExpanded(value => !value)}>
            {expanded ? 'Hide settings' : 'Show settings'} <ChevronDown size={14} className={expanded ? 'flip' : ''} aria-hidden="true" />
          </button>
        </div>
      )}

      {open && (
        <div id={id('fields')} className="form-grid">
          {locked && <p className="lock-note full">Settings are locked so every move in this match uses the same configuration.{keyEditable ? ' You can still replace the API key while paused.' : ''} Reset the match to change them.</p>}
          <Field id={id('provider')} label="Provider" hint={info.help} full>
            <select id={id('provider')} value={player.provider} disabled={locked} aria-describedby={`${id('provider')}-hint`} onChange={event => update('provider', event.target.value)}>
              {PROVIDERS.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </Field>

          {info.needsUrl && (
            <Field id={id('base_url')} label="API base URL" error={errorFor('base_url')} hint="Pick a suggestion or paste the base URL from your provider's docs." full>
              <input
                id={id('base_url')} type="url" inputMode="url" value={player.base_url} disabled={locked}
                list={id('examples')} placeholder="https://api.example.com/v1" autoComplete="off" spellCheck="false"
                aria-invalid={!!errorFor('base_url')} aria-describedby={describedBy('base_url')}
                onChange={event => update('base_url', event.target.value)} onBlur={() => touch('base_url')}
              />
              <datalist id={id('examples')}>
                {info.examples.map(example => <option key={example.url} value={example.url}>{example.label}</option>)}
              </datalist>
            </Field>
          )}

          <Field id={id('model')} label="Model ID" error={errorFor('model')} hint="Exactly as your provider names it.">
            <input
              id={id('model')} value={player.model} disabled={locked} placeholder={info.modelPlaceholder}
              autoComplete="off" spellCheck="false" autoCapitalize="off"
              aria-invalid={!!errorFor('model')} aria-describedby={describedBy('model')}
              onChange={event => update('model', event.target.value)} onBlur={() => touch('model')}
            />
          </Field>

          <Field id={id('api_key')} label="API key" error={errorFor('api_key')} hint="Kept in this tab's memory only.">
            <div className="input-with-button">
              <input
                id={id('api_key')} type={showKey ? 'text' : 'password'} value={player.api_key}
                disabled={locked && !keyEditable} placeholder="Paste key" autoComplete="off" spellCheck="false"
                autoCapitalize="off" data-lpignore="true" data-1p-ignore="true"
                aria-invalid={!!errorFor('api_key')} aria-describedby={describedBy('api_key')}
                onChange={event => update('api_key', event.target.value)} onBlur={() => touch('api_key')}
              />
              <button type="button" className="icon-toggle" aria-label={showKey ? 'Hide API key' : 'Show API key'} aria-pressed={showKey} onClick={() => setShowKey(value => !value)}>
                {showKey ? <EyeOff size={15} aria-hidden="true" /> : <Eye size={15} aria-hidden="true" />}
              </button>
            </div>
          </Field>

          {preview && (
            <p className="endpoint-preview full">
              <span>Requests go to</span> <code>{preview}</code> <span>with</span> <code>{info.auth}</code>
            </p>
          )}

          <Field id={id('preset')} label="Starting preset" hint={preset ? `${preset.max_tokens.toLocaleString()} output tokens · ${preset.timeout_seconds}s per call. ${preset.description}` : 'Choose a starting point or adjust advanced settings. Presets leave reasoning and temperature at the provider default.'} full>
            <select id={id('preset')} value={presetName} disabled={locked} aria-describedby={`${id('preset')}-hint`} onChange={event => onChange(applySettingsPreset(player, event.target.value))}>
              <option value="custom" disabled>Custom settings</option>
              {SETTINGS_PRESETS.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </Field>

          <details className="advanced full">
            <summary>Advanced · reasoning, temperature, token limit, timeout</summary>
            <div className="form-grid inner">
              <Field id={id('reasoning')} label="Reasoning effort" hint={player.reasoning === 'default' ? 'Nothing extra is sent; some models then reason by default and others do not.' : `Sent as ${info.reasoningField}. Not every model accepts it.`}>
                <select id={id('reasoning')} value={player.reasoning} disabled={locked} aria-describedby={`${id('reasoning')}-hint`} onChange={event => update('reasoning', event.target.value)}>
                  {REASONING.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
                </select>
              </Field>
              <Field id={id('temperature')} label="Temperature (optional)" error={errorFor('temperature')} hint={temperatureHint}>
                <input id={id('temperature')} type="number" inputMode="decimal" min="0" max={anthropic ? '1' : '2'} step="any" value={player.temperature ?? ''} disabled={locked} placeholder="Provider default"
                  aria-invalid={!!errorFor('temperature')} aria-describedby={describedBy('temperature')}
                  onChange={event => update('temperature', event.target.value)} onBlur={() => touch('temperature')} />
              </Field>
              <Field id={id('max_tokens')} label="Max output tokens" error={errorFor('max_tokens')} hint="256–64000 per provider request, reasoning included. A cap, not guaranteed usage or a whole-match budget. Provider limits may be lower.">
                <input id={id('max_tokens')} type="number" inputMode="numeric" min="256" max="64000" step="1" value={player.max_tokens} disabled={locked}
                  aria-invalid={!!errorFor('max_tokens')} aria-describedby={describedBy('max_tokens')}
                  onChange={event => update('max_tokens', event.target.value)} onBlur={() => touch('max_tokens')} />
              </Field>
              <Field id={id('timeout_seconds')} label="Timeout (seconds)" error={errorFor('timeout_seconds')} hint="10–600 per provider call. High reasoning can take minutes.">
                <input id={id('timeout_seconds')} type="number" inputMode="numeric" min="10" max="600" step="1" value={player.timeout_seconds} disabled={locked}
                  aria-invalid={!!errorFor('timeout_seconds')} aria-describedby={describedBy('timeout_seconds')}
                  onChange={event => update('timeout_seconds', event.target.value)} onBlur={() => touch('timeout_seconds')} />
              </Field>
              {info.protocol === 'openai_compatible' && (
                <Field id={id('token_parameter')} label="Output token field" hint={info.needsUrl ? 'Automatic sends max_tokens to custom URLs.' : 'Automatic sends max_completion_tokens here.'}>
                  <select id={id('token_parameter')} value={player.token_parameter} disabled={locked} aria-describedby={`${id('token_parameter')}-hint`} onChange={event => update('token_parameter', event.target.value)}>
                    <option value="auto">Automatic</option>
                    <option value="max_tokens">max_tokens</option>
                    <option value="max_completion_tokens">max_completion_tokens</option>
                  </select>
                </Field>
              )}
              <p className="fine-print full">Input tokens come from the complete board, legal moves and game history sent on each turn. Both models receive the same board format and complete history. Token counts vary by tokenizer; reported input and output usage appears with each move. Connection tests and retries are separate billable requests.</p>
            </div>
          </details>

          <div className="verify-row full">
            <button type="button" className="button secondary" onClick={onVerify} disabled={(locked && !keyEditable) || testing}>
              {testing ? <Loader2 size={15} className="spin" aria-hidden="true" /> : <PlugZap size={15} aria-hidden="true" />}
              {testing ? 'Testing…' : 'Test connection'}
            </button>
            <span role="status" className={`verify-message ${verification?.state || ''}`}>
              {verification?.message || 'Asks for one opening move. Not recorded; the provider may bill it.'}
            </span>
          </div>
        </div>
      )}
    </section>
  )
}
