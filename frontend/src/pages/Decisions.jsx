// pages/Decisions.jsx
import { useLiveData } from '../context/LiveDataContext'
import { Card, StatusBadge, EmptyState } from '../components/ui'

const METHOD_COLORS = {
  ai_adaptive: 'text-purple-400',
  emergency_preemption: 'text-red-400',
  density_fallback: 'text-yellow-400',
  rule_fallback: 'text-orange-400',
  fixed_time: 'text-slate-400',
}

export default function Decisions() {
  const { decisions } = useLiveData()

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">AI Decisions</h2>
        <p className="text-sm text-slate-400">{decisions?.length ?? 0} decision(s) recorded</p>
      </div>

      <Card title="Decision Log">
        {!decisions?.length ? (
          <EmptyState message="No decisions recorded yet" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700/50">
                  <th className="text-left py-2 pr-4">Junction</th>
                  <th className="text-left py-2 pr-4">Phase</th>
                  <th className="text-left py-2 pr-4">Method</th>
                  <th className="text-left py-2 pr-4">Safety</th>
                  <th className="text-left py-2 pr-4">Emergency</th>
                  <th className="text-left py-2">Sim Time</th>
                </tr>
              </thead>
              <tbody>
                {decisions.map((d, i) => (
                  <tr key={i} className="border-b border-slate-700/20 hover:bg-slate-800/30">
                    <td className="py-2 pr-4 font-mono text-slate-300">{d.junction ?? '—'}</td>
                    <td className="py-2 pr-4 text-slate-200">{d.selected ?? d.selected_phase ?? '—'}</td>
                    <td className={`py-2 pr-4 font-medium ${METHOD_COLORS[d.method] ?? 'text-slate-400'}`}>
                      {d.method ?? '—'}
                    </td>
                    <td className="py-2 pr-4">
                      {d.safety?.approved === true && <StatusBadge status="APPROVED" />}
                      {d.safety?.approved === false && <StatusBadge status="REJECTED" />}
                      {d.safety?.approved == null && <span className="text-slate-600">—</span>}
                    </td>
                    <td className="py-2 pr-4">
                      {d.emergency_active ? <StatusBadge status="ACTIVE" /> : <span className="text-slate-600">—</span>}
                    </td>
                    <td className="py-2 text-slate-500 font-mono">
                      {(d.time ?? d.simulation_time) != null ? `${Number(d.time ?? d.simulation_time).toFixed(1)}s` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
