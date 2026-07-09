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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
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
