// pages/SystemStatus.jsx
import { useLiveData } from '../context/LiveDataContext'
import { Card, StatusBadge, MetricCard } from '../components/ui'
import { Activity, Brain, Car, Radio } from 'lucide-react'

export default function SystemStatus() {
  const { status, junctions, wsStatus } = useLiveData()

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">System Status</h2>
        <p className="text-sm text-slate-400">Runtime health and configuration overview</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="SUMO Status" value={status?.system_status ?? '—'} icon={Activity}
          accent={status?.system_status === 'running' ? 'green' : 'blue'} />
        <MetricCard label="Sim Time" value={status?.simulation_time != null ? `${Number(status.simulation_time).toFixed(1)}s` : '—'} icon={Activity} accent="blue" />
        <MetricCard label="Junctions" value={junctions?.length ?? 0} icon={Radio} accent="blue" />
        <MetricCard label="Vehicles" value={status?.vehicle_count ?? 0} icon={Car} accent="blue" />
      </div>

      <Card title="System Information">
        <div className="space-y-3 text-sm">
          <Row label="System Status" value={<StatusBadge status={status?.system_status ?? 'stopped'} />} />
          <Row label="Simulation Mode" value={status?.simulation_mode ?? '—'} />
          <Row label="GNN Prediction" value={<StatusBadge status={status?.prediction_available ? 'ACTIVE' : 'INACTIVE'} />} />
          <Row label="WebSocket" value={<StatusBadge status={wsStatus.toUpperCase()} />} />
          <Row label="Started At" value={status?.started_at ? new Date(status.started_at).toLocaleString() : '—'} />
          <Row label="Last Updated" value={status?.updated_at ? new Date(status.updated_at).toLocaleTimeString() : '—'} />
          <Row label="Emergency Count" value={status?.emergency_count ?? 0} />
          <Row label="Current AI Method" value={status?.current_ai_method ?? '—'} />
        </div>
      </Card>

      <Card title="Architecture">
        <pre className="text-xs text-slate-400 whitespace-pre-wrap leading-relaxed">
{`SUMO / TraCI
  → dynamic signalized-junction discovery
  → topology, movements, phase-derived conflict graph
  → valid green phases
  → SUMO traffic state and 10×36 history
  → GNN prediction
  → movement and phase demand
  → near-miss safety risk
  → DecisionEngine
  → emergency priority when present
  → SafetyGate
  → AI / rule / density / fixed-time / safe-stop fallback
  → yellow transition and TraCI phase execution`}
        </pre>
      </Card>
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-700/30 last:border-0">
      <span className="text-slate-400">{label}</span>
      <span className="text-slate-200">{value}</span>
    </div>
  )
}
