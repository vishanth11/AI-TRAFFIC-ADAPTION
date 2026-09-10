// pages/Junctions.jsx
import { useLiveData } from '../context/LiveDataContext'
import { Card, StatusBadge, EmptyState } from '../components/ui'
import { Radio } from 'lucide-react'

export default function Junctions() {
  const { junctions, wsStatus } = useLiveData()

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Live Junctions</h2>
        <p className="text-sm text-slate-400">{junctions?.length ?? 0} junction(s) discovered</p>
      </div>

      {!junctions?.length ? (
        <Card>
          <EmptyState message={wsStatus === 'disconnected' ? 'Backend unavailable' : 'No junctions discovered yet'} />
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {junctions.map(j => <JunctionCard key={j.junction_id} junction={j} />)}
        </div>
      )}
    </div>
  )
}

function JunctionCard({ junction: j }) {
  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-blue-400" />
          <span className="font-semibold text-white">{j.junction_id}</span>
        </div>
        <StatusBadge status={j.safety_approved === true ? 'SAFE' : j.safety_approved === false ? 'REJECTED' : 'UNAVAILABLE'} />
      </div>

      <div className="space-y-1.5 text-sm">
        <Row label="Phase Index" value={j.current_phase_index ?? '—'} />
        <Row label="Phase State" value={<code className="text-xs text-blue-300">{j.current_phase_state ?? '—'}</code>} />
        <Row label="Last Decision" value={j.last_decision ?? '—'} />
        <Row label="Method" value={j.decision_method ?? '—'} />
        <Row label="Remaining" value={j.remaining_time != null ? `${j.remaining_time.toFixed(1)}s` : '—'} />
        <Row label="Sim Time" value={j.simulation_time != null ? `${j.simulation_time.toFixed(1)}s` : '—'} />
        {j.emergency_active && (
          <div className="mt-2 text-xs text-red-400 font-medium bg-red-500/10 rounded px-2 py-1">
            ⚠ Emergency Active {j.emergency_vehicle_id ? `— ${j.emergency_vehicle_id}` : ''}
          </div>
        )}
        {j.safety_checks && (
          <div className="mt-2 pt-2 border-t border-slate-700/50">
            <p className="text-xs text-slate-500 mb-1">Safety Checks</p>
            {Object.entries(j.safety_checks).map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs">
                <span className="text-slate-400">{k}</span>
                <span className={v ? 'text-emerald-400' : 'text-red-400'}>{v ? '✓' : '✗'}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="text-slate-200">{value}</span>
    </div>
  )
}
