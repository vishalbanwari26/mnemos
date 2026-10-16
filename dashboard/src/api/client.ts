import type {
  ChatResponse,
  Episode,
  MemoryQueryResult,
  MemoryStats,
  ProceduralStrategyStats,
  ReflectionLogEntry,
  ReflectionRunResult,
  SemanticFact,
} from './types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Static GitHub Pages demo (VITE_STATIC=1): every read comes from a recording of
// the real API (scripts/record_demo.py) for one fictional user, and the chat and
// retrieval trace answer the recorded questions.
export const STATIC_DEMO = import.meta.env.VITE_STATIC === '1'
const DEMO_DATA = `${import.meta.env.BASE_URL}demo-data/`

export interface DemoPresets {
  user: string
  chats: { message: string; response: ChatResponse }[]
  traces: { query: string; result: MemoryQueryResult }[]
}
let presets: Promise<DemoPresets> | null = null
export const loadPresets = () =>
  (presets ??= fetch(`${DEMO_DATA}presets.json`).then((r) => r.json() as Promise<DemoPresets>))

const normalise = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim()
const ONLY_PRESETS =
  'This recorded demo answers the suggested questions only. Run mnemos locally to ask anything.'

async function recorded<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? 'GET'
  if (method === 'GET') {
    const [p, q = ''] = path.replace(/^\/users\/[^/]+\//, '').split('?')
    const suffix = q.split('&').filter(Boolean).map((kv) => '__' + kv.replace('=', '-')).join('')
    const res = await fetch(`${DEMO_DATA}${p}${suffix}.json`)
    if (!res.ok) throw new Error(`not in the recorded demo (${res.status})`)
    return res.json() as Promise<T>
  }
  const body = JSON.parse((init?.body as string) ?? '{}')
  const { chats, traces } = await loadPresets()
  await new Promise((r) => setTimeout(r, 450))
  if (path.endsWith('/messages')) {
    const hit = chats.find((c) => normalise(c.message) === normalise(body.message ?? ''))
    if (hit) return hit.response as T
  }
  if (path.endsWith('/retrieval-trace')) {
    const hit = traces.find((t) => normalise(t.query) === normalise(body.query ?? ''))
    if (hit) return hit.result as T
  }
  throw new Error(path.endsWith('/messages') || path.endsWith('/retrieval-trace') ? ONLY_PRESETS
    : 'Changing memory is not available in the recorded demo. Run mnemos locally to try it.')
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (STATIC_DEMO) return recorded<T>(path, init)
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export const api = {
  sendMessage: (userId: string, message: string, sessionId?: string) =>
    request<ChatResponse>(`/users/${userId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId }),
    }),

  resetUser: (userId: string) =>
    request<void>(`/users/${userId}/reset`, { method: 'POST' }),

  listEpisodic: (userId: string) =>
    request<{ episodes: Episode[] }>(`/users/${userId}/memories/episodic`),

  listSemantic: (userId: string, status = 'active') =>
    request<{ facts: SemanticFact[] }>(
      `/users/${userId}/memories/semantic?status=${status}`,
    ),

  forgetFact: (userId: string, factId: string, reason: string) =>
    request<void>(
      `/users/${userId}/memories/semantic/${factId}?reason=${encodeURIComponent(reason)}`,
      { method: 'DELETE' },
    ),

  retrievalTrace: (userId: string, query: string, asOf?: string) =>
    request<MemoryQueryResult>(`/users/${userId}/retrieval-trace`, {
      method: 'POST',
      body: JSON.stringify({ query, as_of: asOf || null }),
    }),

  getStats: (userId: string) => request<MemoryStats>(`/users/${userId}/stats`),

  runReflection: (userId: string) =>
    request<ReflectionRunResult>(`/users/${userId}/reflect`, { method: 'POST' }),

  getReflectionLog: (userId: string) =>
    request<{ entries: ReflectionLogEntry[] }>(`/users/${userId}/reflection-log`),

  getProceduralStats: (userId: string) =>
    request<{ strategies: ProceduralStrategyStats[] }>(`/users/${userId}/procedural`),
}
