import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { useUser } from '../context/UserContext'

export default function MemoryBrowserPage() {
  const { userId } = useUser()
  const [status, setStatus] = useState<'active' | 'merged' | 'forgotten'>('active')
  const queryClient = useQueryClient()

  const episodic = useQuery({
    queryKey: ['episodic', userId],
    queryFn: () => api.listEpisodic(userId),
  })
  const semantic = useQuery({
    queryKey: ['semantic', userId, status],
    queryFn: () => api.listSemantic(userId, status),
  })

  const forget = useMutation({
    mutationFn: (factId: string) => api.forgetFact(userId, factId, 'user_requested'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['semantic', userId] })
      queryClient.invalidateQueries({ queryKey: ['stats', userId] })
    },
  })

  return (
    <>
      <h2 className="page-title">Memory Browser</h2>

      <div className="card">
        <h3>Semantic facts</h3>
        <div className="filter-row">
          {(['active', 'merged', 'forgotten'] as const).map((s) => (
            <button
              key={s}
              className={status === s ? 'primary' : ''}
              onClick={() => setStatus(s)}
            >
              {s}
            </button>
          ))}
        </div>
        {semantic.isLoading && <p className="muted">Loading…</p>}
        {semantic.data?.facts.length === 0 && (
          <p className="muted">No {status} facts.</p>
        )}
        {semantic.data?.facts.map((f) => (
          <div className="list-row" key={f.id}>
            <span>
              <span className={`badge ${f.status}`}>{f.status}</span> {f.fact}
            </span>
            <span className="meta">
              confidence {f.confidence.toFixed(2)}
              {status === 'active' && (
                <button
                  style={{ marginLeft: 10 }}
                  onClick={() => forget.mutate(f.id)}
                  disabled={forget.isPending}
                >
                  forget
                </button>
              )}
            </span>
          </div>
        ))}
      </div>

      <div className="card">
        <h3>Episodic memory</h3>
        {episodic.isLoading && <p className="muted">Loading…</p>}
        {episodic.data?.episodes.length === 0 && (
          <p className="muted">No episodes yet.</p>
        )}
        {episodic.data?.episodes.map((e) => (
          <div className="list-row" key={e.id}>
            <span>
              <strong>{e.role}</strong> {e.content}
            </span>
            <span className="meta">
              {new Date(e.occurred_at).toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </>
  )
}
