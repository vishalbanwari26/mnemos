import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { useUser } from '../context/UserContext'

const ALL_STRATEGIES = ['semantic_heavy', 'balanced', 'recency_heavy']

export default function ProceduralPage() {
  const { userId } = useUser()
  const stats = useQuery({
    queryKey: ['procedural', userId],
    queryFn: () => api.getProceduralStats(userId),
  })

  const byName = new Map(stats.data?.strategies.map((s) => [s.strategy_name, s]))

  return (
    <>
      <h2 className="page-title">Procedural Strategies</h2>
      <p className="muted">
        mnemos tracks which similarity/recency weighting works best per user
        (epsilon-greedy over an empirical success rate — a message is scored
        as a "failure" if the user's next reply looks like a correction).
        This is what the agent is adapting toward.
      </p>

      {stats.isLoading && <p className="muted">Loading…</p>}

      <div className="card">
        {ALL_STRATEGIES.map((name) => {
          const s = byName.get(name)
          const uses = s?.uses ?? 0
          const rate = s?.success_rate ?? 0
          return (
            <div key={name} style={{ marginBottom: 14 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 13,
                  marginBottom: 4,
                }}
              >
                <span>{name}</span>
                <span className="muted">
                  {uses === 0 ? 'not yet used' : `${(rate * 100).toFixed(0)}% success · ${uses} uses`}
                </span>
              </div>
              <div
                style={{
                  background: 'var(--surface-2)',
                  borderRadius: 4,
                  height: 8,
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${uses === 0 ? 0 : rate * 100}%`,
                    height: '100%',
                    background: 'var(--series-episodic)',
                    borderRadius: 4,
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}
