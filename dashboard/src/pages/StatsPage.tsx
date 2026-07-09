import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { useUser } from '../context/UserContext'
import GrowthChart from '../components/GrowthChart'

export default function StatsPage() {
  const { userId } = useUser()
  const stats = useQuery({
    queryKey: ['stats', userId],
    queryFn: () => api.getStats(userId),
  })

  if (stats.isLoading) return <p className="muted">Loading…</p>
  if (stats.isError) {
    return <div className="error-banner">{(stats.error as Error).message}</div>
  }
  if (!stats.data) return null

  const d = stats.data

  return (
    <>
      <h2 className="page-title">Timeline &amp; Stats</h2>

      <div className="card stat-row">
        <div className="stat-tile">
          <div className="value">{d.episodic_total}</div>
          <div className="label">Episodic</div>
        </div>
        <div className="stat-tile">
          <div className="value">{d.semantic_active}</div>
          <div className="label">Semantic (active)</div>
        </div>
        <div className="stat-tile">
          <div className="value">{d.semantic_merged}</div>
          <div className="label">Merged</div>
        </div>
        <div className="stat-tile">
          <div className="value">{d.semantic_forgotten}</div>
          <div className="label">Forgotten</div>
        </div>
      </div>

      <div className="card">
        <h3>Growth over time</h3>
        <GrowthChart
          series={[
            { name: 'Episodic', color: 'var(--series-episodic)', data: d.episodic_by_day },
            { name: 'Semantic', color: 'var(--series-semantic)', data: d.semantic_by_day },
          ]}
        />
      </div>
    </>
  )
}
