// pages/Logs.jsx
import { useLiveData } from '../context/LiveDataContext'
import { Card, EmptyState } from '../components/ui'

const EVENT_COLORS = {
  system_start: 'text-blue-400',
  system_complete: 'text-emerald-400',
  decision: 'text-purple-400',
  error: 'text-red-400',
}

export default function Logs() {
  const { logs } = useLiveData()

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">System Logs</h2>
        <p className="text-sm text-slate-400">{logs?.length ?? 0} log entries (most recent first)</p>
      </div>

      <Card title="Runtime Log">
        {!logs?.length ? (
          <EmptyState message="No log entries yet" />
        ) : (
          <div className="space-y-0 font-mono text-xs max-h-[70vh] overflow-y-auto">
            {logs.map((log, i) => (
              <div key={i} className="flex gap-3 py-1.5 border-b border-slate-700/20 last:border-0 hover:bg-slate-800/20">
                <span className="text-slate-600 w-20 flex-shrink-0">
                  {log.simulation_time != null ? `${Number(log.simulation_time).toFixed(1)}s` : '—'}
                </span>
                <span className={`w-28 flex-shrink-0 font-medium ${EVENT_COLORS[log.event] ?? 'text-slate-400'}`}>
                  {log.event}
                </span>
                <span className="text-slate-400 flex-1 truncate">
                  {formatLogFields(log)}
                </span>
                <span className="text-slate-700 flex-shrink-0 hidden lg:block">
                  {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : ''}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

function formatLogFields(log) {
  const skip = new Set(['timestamp', 'simulation_time', 'event'])
  const parts = Object.entries(log)
    .filter(([k]) => !skip.has(k))
    .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`)
  return parts.join('  ') || '—'
}
