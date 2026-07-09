import type { DailyCount } from '../api/types'

interface Series {
  name: string
  color: string
  data: DailyCount[]
}

export default function GrowthChart({ series }: { series: Series[] }) {
  const allDates = Array.from(
    new Set(series.flatMap((s) => s.data.map((d) => d.date))),
  ).sort()

  if (allDates.length === 0) {
    return <p className="muted">No data yet.</p>
  }

  const maxCount = Math.max(
    1,
    ...series.flatMap((s) => s.data.map((d) => d.count)),
  )

  const width = 640
  const height = 180
  const padding = { top: 8, right: 8, bottom: 24, left: 8 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const groupWidth = plotWidth / allDates.length
  const barWidth = Math.max(2, groupWidth / (series.length + 1) - 2)

  return (
    <div className="viz-root">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Memory growth over time"
        style={{ width: '100%', height: 'auto' }}
      >
        {/* recessive baseline */}
        <line
          x1={padding.left}
          y1={height - padding.bottom}
          x2={width - padding.right}
          y2={height - padding.bottom}
          stroke="var(--border)"
          strokeWidth={1}
        />
        {allDates.map((date, i) => {
          const groupX = padding.left + i * groupWidth
          return (
            <g key={date}>
              {series.map((s, si) => {
                const count = s.data.find((d) => d.date === date)?.count ?? 0
                const barHeight = (count / maxCount) * plotHeight
                const x = groupX + si * (barWidth + 2)
                const y = height - padding.bottom - barHeight
                return (
                  <rect
                    key={s.name}
                    x={x}
                    y={y}
                    width={barWidth}
                    height={Math.max(barHeight, count > 0 ? 2 : 0)}
                    rx={2}
                    fill={s.color}
                  >
                    <title>
                      {s.name}: {count} on {date}
                    </title>
                  </rect>
                )
              })}
              {(i === 0 || i === allDates.length - 1) && (
                <text
                  x={groupX}
                  y={height - 8}
                  fontSize={10}
                  fill="var(--text-muted)"
                >
                  {date}
                </text>
              )}
            </g>
          )
        })}
      </svg>
      <div style={{ display: 'flex', gap: 16, marginTop: 4 }}>
        {series.map((s) => (
          <span
            key={s.name}
            className="muted"
            style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: s.color,
                display: 'inline-block',
              }}
            />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  )
}
