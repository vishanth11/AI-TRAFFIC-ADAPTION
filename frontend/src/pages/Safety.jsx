// pages/Safety.jsx
import { useLiveData } from '../context/LiveDataContext'
import { Card, StatusBadge, MetricCard, EmptyState, CheckRow } from '../components/ui'
import { ShieldAlert, Siren } from 'lucide-react'

export default function Safety() {
  const { safety, emergencies } = useLiveData()

  const riskLevel = safety?.risk_level ?? 'UNKNOWN'
  const riskScore = safety?.risk_score ?? 0
  const nearMissRisk = safety?.near_miss_risk ?? 0
  const confidence = safety?.safety_confidence ?? 0
  const pairRisks = safety?.pair_risks ?? []

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Safety & Emergency</h2>
        <p className="text-sm text-slate-400">Real-time safety assessment and emergency vehicle tracking</p>
      </div>

      {/* Risk summary */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Risk Level" value={riskLevel} icon={ShieldAlert}
          accent={riskLevel === 'CRITICAL' ? 'red' : riskLevel === 'HIGH' ? 'yellow' : riskLevel === 'MEDIUM' ? 'yellow' : 'green'} />
        <MetricCard label="Risk Score" value={riskScore.toFixed(3)} icon={ShieldAlert} accent="blue" />
        <MetricCard label="Near-Miss Risk" value={nearMissRisk.toFixed(3)} icon={ShieldAlert} accent="purple" />
        <MetricCard label="Safety Confidence" value={confidence.toFixed(3)} icon={ShieldAlert} accent="green" />
      </div>

      {/* Accident status */}
      <Card title="Accident Detection">
        <div className="flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-slate-500 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm text-slate-300 font-medium">Status</p>
            {safety?.accident_status?.includes('unavailable') ? (
              <>
                <p className="text-sm font-semibold text-slate-400 mt-1">Unavailable</p>
                <p className="text-xs text-slate-500 mt-1">
                  Reason: No live camera frame available in SUMO telemetry mode.
                </p>
              </>
            ) : (
              <p className="text-xs text-slate-400 mt-1">{safety?.accident_status ?? 'unavailable_no_camera_frame_or_model'}</p>
            )}
          </div>
        </div>
      </Card>

      {/* Pair risks */}
      <Card title="Vehicle Pair Risks">
        {!pairRisks.length ? (
          <EmptyState message="No pair risk data available" />
        ) : (
          <div className="space-y-2">
            {pairRisks.map((p, i) => (
              <div key={i} className="flex items-center justify-between text-sm py-1.5 border-b border-slate-700/30 last:border-0">
                <span className="text-slate-300 font-mono">
                  {Array.isArray(p.vehicles) ? p.vehicles.join(' ↔ ') : `Pair ${i + 1}`}
                </span>
                <div className="flex items-center gap-3">
                  {p.ttc_seconds != null && <span className="text-slate-400">TTC: <span className="text-white">{Number(p.ttc_seconds).toFixed(2)}s</span></span>}
                  {p.risk_score != null && <span className="text-slate-400">Risk: <span className="text-white">{Number(p.risk_score).toFixed(3)}</span></span>}
                  {p.near_miss_probability != null && <span className="text-slate-400">NM: <span className="text-white">{Number(p.near_miss_probability).toFixed(3)}</span></span>}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Safety checks from latest assessment */}
      {safety?.assessment?.checks && (
        <Card title="Safety Gate Checks">
          {Object.entries(safety.assessment.checks).map(([k, v]) => (
            <CheckRow key={k} label={k} value={v} />
          ))}
        </Card>
      )}

      {/* Emergency vehicles */}
      <Card title="Emergency Vehicles">
        {!emergencies?.length ? (
          <EmptyState message="No emergency vehicles active" />
        ) : (
          <div className="space-y-3">
            {emergencies.map((ev, i) => (
              <EmergencyCard key={ev.vehicle_id ?? i} ev={ev} />
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

function EmergencyCard({ ev }) {
  return (
    <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3">
      <div className="flex items-center gap-2 mb-2">
        <Siren className="w-4 h-4 text-red-400" />
        <span className="font-semibold text-red-300">{ev.vehicle_id ?? 'Unknown'}</span>
        <StatusBadge status="ACTIVE" />
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400">
        {ev.edge && <span>Edge: <span className="text-slate-200">{ev.edge}</span></span>}
        {ev.speed != null && <span>Speed: <span className="text-slate-200">{Number(ev.speed).toFixed(1)} m/s</span></span>}
        {ev.distance != null && <span>Distance: <span className="text-slate-200">{Number(ev.distance).toFixed(1)} m</span></span>}
        {ev.eta != null && <span>ETA: <span className="text-slate-200">{Number(ev.eta).toFixed(1)}s</span></span>}
        {ev.current_junction && <span>Junction: <span className="text-slate-200">{ev.current_junction}</span></span>}
        {ev.confidence != null && <span>Confidence: <span className="text-slate-200">{Number(ev.confidence).toFixed(2)}</span></span>}
      </div>
    </div>
  )
}
