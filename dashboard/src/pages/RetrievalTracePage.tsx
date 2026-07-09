import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import { useUser } from '../context/UserContext'

function scoreBar(score: number, color: string) {
  const pct = Math.max(0, Math.min(1, score)) * 100
  return (
    <div
      style={{
        background: 'var(--surface-2)',
        borderRadius: 4,
        height: 6,
        width: 120,
        overflow: 'hidden',
      }}
    >
      <div style={{ background: color, width: `${pct}%`, height: '100%' }} />
    </div>
  )
}

export default function RetrievalTracePage() {
  const { userId } = useUser()
  const [query, setQuery] = useState('')
  const [asOf, setAsOf] = useState('')

  const trace = useMutation({
    mutationFn: () => api.retrievalTrace(userId, query, asOf || undefined),
  })

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (query.trim()) trace.mutate()
  }

  return (
    <>
      <h2 className="page-title">Retrieval Trace</h2>
      <p className="muted">
        Runs the same recall the chat loop uses, without calling the LLM —
        see exactly what would be retrieved and why. Set an "as of" date to
        time-travel: score recency as though this query were asked back then.
      </p>

      <form className="card" onSubmit={handleSubmit}>
        <div className="filter-row" style={{ alignItems: 'center' }}>
          <input
            style={{ flex: 1 }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What backend framework do I use?"
          />
          <label className="muted" style={{ whiteSpace: 'nowrap' }}>
            as of{' '}
            <input
              type="datetime-local"
              value={asOf}
              onChange={(e) => setAsOf(e.target.value)}
            />
          </label>
          <button type="submit" className="primary" disabled={trace.isPending}>
            Trace
          </button>
        </div>
      </form>

      {trace.isError && (
        <div className="error-banner">{(trace.error as Error).message}</div>
      )}

      {trace.data && (
        <>
          <div className="card">
            <h3>Semantic facts ({trace.data.facts.length})</h3>
            {trace.data.facts.length === 0 && (
              <p className="muted">Nothing retrieved.</p>
            )}
            {trace.data.facts.map((sf) => (
              <div className="list-row" key={sf.fact.id}>
                <span>{sf.fact.fact}</span>
                <span
                  className="meta"
                  style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                >
                  similarity {sf.similarity.toFixed(2)}
                  {scoreBar(sf.similarity, 'var(--series-semantic)')}
                </span>
              </div>
            ))}
          </div>

          <div className="card">
            <h3>Episodic memory ({trace.data.episodes.length})</h3>
            {trace.data.episodes.length === 0 && (
              <p className="muted">Nothing retrieved.</p>
            )}
            {trace.data.episodes.map((se) => (
              <div className="list-row" key={se.episode.id}>
                <span>
                  <strong>{se.episode.role}</strong> {se.episode.content}
                </span>
                <span
                  className="meta"
                  style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                >
                  sim {se.similarity.toFixed(2)} · recency{' '}
                  {se.recency_factor.toFixed(2)} · score {se.score.toFixed(2)}
                  {scoreBar(se.score, 'var(--series-episodic)')}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  )
}
