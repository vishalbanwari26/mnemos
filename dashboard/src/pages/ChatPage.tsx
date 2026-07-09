import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { useUser } from '../context/UserContext'

interface Turn {
  role: 'user' | 'assistant'
  content: string
  meta?: string
}

export default function ChatPage() {
  const { userId } = useUser()
  const [turns, setTurns] = useState<Turn[]>([])
  const [message, setMessage] = useState('')
  const [sessionId, setSessionId] = useState<string | undefined>(undefined)
  const queryClient = useQueryClient()

  const sendMessage = useMutation({
    mutationFn: (text: string) => api.sendMessage(userId, text, sessionId),
    onSuccess: (res) => {
      setSessionId(res.session_id)
      setTurns((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.reply,
          meta: `${res.memory_used} memories used · strategy: ${res.strategy_used}${
            res.facts_learned.length
              ? ` · learned: ${res.facts_learned.join(', ')}`
              : ''
          }`,
        },
      ])
      queryClient.invalidateQueries({ queryKey: ['stats', userId] })
      queryClient.invalidateQueries({ queryKey: ['episodic', userId] })
      queryClient.invalidateQueries({ queryKey: ['semantic', userId] })
      queryClient.invalidateQueries({ queryKey: ['procedural', userId] })
    },
  })

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const text = message.trim()
    if (!text) return
    setTurns((prev) => [...prev, { role: 'user', content: text }])
    setMessage('')
    sendMessage.mutate(text)
  }

  function handleReset() {
    setTurns([])
    setSessionId(undefined)
  }

  return (
    <>
      <h2 className="page-title">Chat</h2>
      <p className="muted">
        Talking to <strong>{userId}</strong>. Restart the conversation below,
        or switch users in the sidebar to test cross-session recall.
      </p>

      <div className="card">
        <div className="chat-log">
          {turns.length === 0 && (
            <p className="muted">Say something — mnemos will remember it.</p>
          )}
          {turns.map((t, i) => (
            <div key={i} className={`chat-turn ${t.role}`}>
              {t.content}
              {t.meta && <span className="meta">{t.meta}</span>}
            </div>
          ))}
          {sendMessage.isPending && <div className="chat-turn assistant">…</div>}
        </div>

        {sendMessage.isError && (
          <div className="error-banner">{(sendMessage.error as Error).message}</div>
        )}

        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type a message…"
            disabled={sendMessage.isPending}
          />
          <button type="submit" className="primary" disabled={sendMessage.isPending}>
            Send
          </button>
          <button type="button" onClick={handleReset}>
            New session
          </button>
        </form>
      </div>
    </>
  )
}
