// pages/Dashboard.jsx
import { useLiveData } from '../context/LiveDataContext'
import { MetricCard, Card, StatusBadge, EmptyState } from '../components/ui'
import { Radio, Car, Siren, ShieldAlert, Brain, Activity } from 'lucide-react'

function methodLabel(method) {
  if (!method) return null
  const map = {
    ai_adaptive: 'AI ADAPTIVE',
    emergency_preemption: 'EMERGENCY',
    density_fallback: 'DENSITY FALLBACK',
    rule_fallback: 'RULE FALLBACK',
    fixed_time: 'FIXED TIME',
  }
  return map[method] ?? method.toUpperCase()
}

function riskAccent(level) {
  if (level === 'CRITICAL') return 'red'
  if (level === 'HIGH') return 'yellow'
  if (level === 'MEDIUM') return 'yellow'
  return 'green'
}

export default function Dashboard() {
  const { status, junctions, decisions, emergencies, safety, wsStatus } = useLiveData()

  const riskLevel = safety?.risk_level ?? status?.risk_level ?? 'UNKNOWN'
  const method = methodLabel(status?.current_ai_method)
  const recentDecisions = decisions?.slice(0, 5) ?? []

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Dashboard</h2>
        <p className="text-sm text-slate-400">Real-time traffic system overview</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          label="Active Junctions"
          value={junctions?.length ?? '—'}
          icon={Radio}
          accent="blue"
        />
        <MetricCard
          label="Active Vehicles"
          value={status?.vehicle_count ?? '—'}
          icon={Car}
          accent="blue"
        />
        <MetricCard
          label="Emergencies"
          value={emergencies?.length ?? 0}
          icon={Siren}
          accent={emergencies?.length > 0 ? 'red' : 'green'}
        />
        <MetricCard
          label="Current Risk"
          value={riskLevel}
          icon={ShieldAlert}
          accent={riskAccent(riskLevel)}
        />
        <MetricCard
          label="AI Method"
          value={method ?? '—'}
          icon={Brain}
          accent="purple"
        />
      </div>

      {/* Junction overview grid */}
      <Card title="Junction Overview">
        {!junctions?.length ? (
          <EmptyState message={wsStatus === 'disconnected' ? 'Backend unavailable' : 'No junctions discovered yet'} />
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {junctions.map(j => (
              <JunctionMiniCard key={j.junction_id} junction={j} />
            ))}
          </div>
        )}
      </Card>

      {/* Recent decisions */}
      <Card title="Recent AI Decisions">
        {!recentDecisions.length ? (
          <EmptyState message="No decisions recorded yet" />
        ) : (
          <div className="space-y-2">
            {recentDecisions.map((d, i) => (
              <DecisionRow key={i} decision={d} />
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

function JunctionMiniCard({ junction }) {
  const safe = junction.safety_approved
  return (
    <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold text-white">{junction.junction_id}</span>
        {safe === true && <StatusBadge status="SAFE" />}
        {safe === false && <StatusBadge status="REJECTED" />}
        {safe == null && <StatusBadge status="UNAVAILABLE" />}
      </div>
      <div className="space-y-1 text-xs text-slate-400">
        <div className="flex justify-between">
          <span>Phase</span>
          <span className="text-slate-300">{junction.last_decision ?? '—'}</span>
        </div>
        <div className="flex justify-between">
          <span>Method</span>
          <span className="text-slate-300">{junction.decision_method ?? '—'}</span>
        </div>
        <div className="flex justify-between">
          <span>Remaining</span>
          <span className="text-slate-300">
            {junction.remaining_time != null ? `${junction.remaining_time.toFixed(0)}s` : '—'}
          </span>
        </div>
        {junction.emergency_active && (
          <div className="text-red-400 font-medium">⚠ Emergency Active</div>
        )}
      </div>
    </div>
  )
}

function DecisionRow({ decision }) {
  const approved = decision.safety?.approved
  return (
    <div className="flex items-center gap-3 py-2 border-b border-slate-700/30 last:border-0 text-xs">
      <span className="text-slate-400 font-mono w-12">{decision.junction ?? '—'}</span>
      <span className="text-slate-300 flex-1">{decision.selected ?? decision.selected_phase ?? '—'}</span>
      <span className="text-slate-500">{decision.method ?? '—'}</span>
      {approved === true && <StatusBadge status="APPROVED" />}
      {approved === false && <StatusBadge status="REJECTED" />}
      {decision.emergency_active && <StatusBadge status="ACTIVE" />}
    </div>
  )
}
