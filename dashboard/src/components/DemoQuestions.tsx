import { useEffect, useState } from 'react'
import { loadPresets, type DemoPresets } from '../api/client'

/** Clickable questions for the static demo, taken from the recording. */
export default function DemoQuestions({ pick, onPick, disabled }: {
  pick: (p: DemoPresets) => string[]
  onPick: (q: string) => void
  disabled?: boolean
}) {
  const [questions, setQuestions] = useState<string[]>([])
  // Load once: `pick` is a new function on every render of the parent.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { loadPresets().then((p) => setQuestions(pick(p))) }, [])
  return (
    <div className="demo-questions">
      <span className="muted">Try a recorded question:</span>
      {questions.map((q) => (
        <button key={q} type="button" onClick={() => onPick(q)} disabled={disabled}>{q}</button>
      ))}
    </div>
  )
}
