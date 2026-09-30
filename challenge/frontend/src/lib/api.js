import { requestPlayer } from './providers.js'

export const API_BASE = (import.meta.env?.VITE_CHALLENGE_API_BASE_URL || '/api').replace(/\/$/, '')

export function redactKey(message, key) {
  const secret = key?.trim()
  return secret ? message.replaceAll(secret, '[REDACTED API KEY]') : message
}

export class MoveError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'MoveError'
    this.status = status
  }

  get kind() {
    if (this.status === 0) return 'network'
    if (this.status === 401) return 'auth'
    if (this.status === 429) return 'rate'
    if (this.status === 503 || this.status === 504) return 'busy'
    if (this.status === 409) return 'mismatch'
    if (this.status === 400 || this.status === 422) return 'config'
    return 'provider'
  }
}

export async function requestMove(player, moves, signal) {
  let response
  try {
    response = await fetch(`${API_BASE}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player: requestPlayer(player), moves }),
      signal,
      cache: 'no-store',
      credentials: 'same-origin',
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new MoveError('Could not reach the challenge server. Check your connection or that the server is running.', 0)
  }
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = typeof data?.detail === 'string' ? redactKey(data.detail, player.api_key) : `Request failed (HTTP ${response.status})`
    throw new MoveError(detail, response.status)
  }
  if (!data || typeof data !== 'object') throw new MoveError('The challenge server returned an unreadable response.', 502)
  return data
}
