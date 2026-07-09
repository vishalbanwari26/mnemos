import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { useUser } from '../context/UserContext'

const ACTION_LABEL: Record<string, string> = {
  merge: 'Merged',
  decay: 'Decayed',
  forget: 'Forgotten',
}

export default function ReflectionLogPage() {
  const { userId } = useUser()
  const queryClient = useQueryClient()

  const log = useQuery({
    queryKey: ['reflection-log', userId],
    queryFn: () => api.getReflectionLog(userId),
  })

  const runReflection = useMutation({
    mutationFn: () => api.runReflection(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reflection-log', userId] })
      queryClient.invalidateQueries({ queryKey: ['semantic', userId] })
      queryClient.invalidateQueries({ queryKey: ['stats', userId] })
    },
  })

  return (
    <>
      <h2 className="page-title">Reflection Log</h2>
      <p className="muted">
        Merges near-duplicate facts, decays confidence on facts nobody has
        needed in a while, and archives anything that decays past the forget
        threshold. Triggered on demand — there's no background scheduler.
      </p>

      <div className="card">
        <button
          className="primary"
          onClick={() => runReflection.mutate()}
          disabled={runReflection.isPending}
        >
          {runReflection.isPending ? 'Running…' : 'Run reflection now'}
        </button>
        {runReflection.data && (
          <p className="muted" style={{ marginTop: 10 }}>
            Merged into {runReflection.data.facts_merged_into} fact(s), decayed{' '}
            {runReflection.data.facts_decayed}, forgot{' '}
            {runReflection.data.facts_forgotten}.
          </p>
        )}
        {runReflection.isError && (
          <div className="error-banner">
            {(runReflection.error as Error).message}
          </div>
        )}
      </div>

      <div className="card">
        <h3>Audit trail</h3>
        {log.isLoading && <p className="muted">Loading…</p>}
        {log.data?.entries.length === 0 && (
          <p className="muted">No reflection actions yet.</p>
        )}
        {log.data?.entries.map((entry) => (
          <div className="list-row" key={entry.id}>
            <span>
              <span className={`badge ${entry.action === 'forget' ? 'forgotten' : entry.action === 'merge' ? 'merged' : 'active'}`}>
                {ACTION_LABEL[entry.action] ?? entry.action}
              </span>{' '}
              {entry.detail}
            </span>
            <span className="meta">
              {new Date(entry.created_at).toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </>
  )
}
