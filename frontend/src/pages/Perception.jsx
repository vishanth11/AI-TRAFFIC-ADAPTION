// pages/Perception.jsx
import { useState, useEffect } from 'react'
import { useLiveData } from '../context/LiveDataContext'
import { Card, MetricCard, EmptyState, DataSourceTag } from '../components/ui'
import { Eye, Activity, AlertTriangle } from 'lucide-react'
import { fetchComparison } from '../services/api'

export default function Perception() {
  const { safety, status } = useLiveData()
  const [comparison, setComparison] = useState([])

  useEffect(() => {
    fetchComparison().then(setComparison).catch(() => {})
  }, [])

  const riskScore = safety?.risk_score ?? 0
  const nearMissRisk = safety?.near_miss_risk ?? 0
  const confidence = safety?.safety_confidence ?? 0
  const pairRisks = safety?.pair_risks ?? []
  const accidentStatus = safety?.accident_status ?? 'unavailable_no_camera_frame_or_model'
  const isAccidentUnavailable = accidentStatus.includes('unavailable')

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Perception & Risk</h2>
        <p className="text-sm text-slate-400">Near-miss risk, safety intelligence, and perception pipeline status</p>
      </div>

      {/* Risk scores — from SUMO telemetry */}
      <div className="flex items-center gap-2 mb-1">
        <DataSourceTag source="SUMO / TraCI" />
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard label="Overall Risk Score" value={riskScore.toFixed(4)} icon={Activity} accent="red" />
        <MetricCard label="Near-Miss Risk" value={nearMissRisk.toFixed(4)} icon={Eye} accent="yellow" />
        <MetricCard label="Model Confidence" value={confidence.toFixed(4)} icon={Eye} accent="blue" />
      </div>

      {/* Accident detection — clearly marked unavailable */}
      <Card title="Accident Detection">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-slate-500 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <p className="text-sm text-slate-300 font-medium">Image-based Accident Detector</p>
              <DataSourceTag source="Computer Vision" />
            </div>
            {isAccidentUnavailable ? (
              <>
                <p className="text-sm font-semibold text-slate-400 mt-2">Unavailable</p>
                <p className="text-xs text-slate-500 mt-1">
                  Reason: No live camera frame available in SUMO telemetry mode.
                </p>
                <p className="text-xs text-slate-600 mt-1">
                  The calibrated accident detector requires image input. SUMO does not provide camera frames.
                  Accident inference remains explicitly unavailable until a real image provider is connected.
                </p>
              </>
            ) : (
              <p className="text-xs text-slate-400 mt-1">{accidentStatus}</p>
            )}
          </div>
        </div>
      </Card>

      {/* YOLO perception pipeline — separate source */}
      <Card title="Computer Vision Pipeline">
        <div className="flex items-center gap-2 mb-3">
          <DataSourceTag source="Computer Vision / YOLO" />
        </div>
        <div className="text-xs text-slate-500 space-y-1">
          <p>The YOLO-based perception pipeline (detect_traffic.py, detect_emergency.py, track_vehicles.py) runs separately from the SUMO simulation.</p>
          <p className="mt-2 text-slate-600">Output is written to <code className="text-slate-400">outputs/perception_output.json</code>.</p>
          <p className="mt-2 text-yellow-600 font-medium">⚠ YOLO perception data is NOT live SUMO telemetry. Do not merge these sources.</p>
        </div>
      </Card>

      {/* AI method */}
      <Card title="Current AI Control Method">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-purple-400" />
          <div>
            <p className="text-sm text-slate-300 font-medium">Active Method</p>
            <p className="text-lg font-bold text-purple-300 mt-1">
              {status?.current_ai_method ?? '—'}
            </p>
          </div>
          <DataSourceTag source="SUMO / TraCI" />
        </div>
        <p className="text-xs text-slate-500 mt-3">
          Fallback chain: ai_adaptive → rule_fallback → density_fallback → fixed_time → safe_stop
        </p>
      </Card>

      {/* Vehicle pair risks */}
      <Card title="Vehicle Pair Risk Details">
        <div className="flex items-center gap-2 mb-3">
          <DataSourceTag source="SUMO / TraCI" />
        </div>
        {!pairRisks.length ? (
          <EmptyState message="No pair risk data — simulation may not be running" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700/50">
                  <th className="text-left py-2 pr-4">Vehicles</th>
                  <th className="text-left py-2 pr-4">TTC (s)</th>
                  <th className="text-left py-2 pr-4">Near-Miss P</th>
                  <th className="text-left py-2">Risk Score</th>
                </tr>
              </thead>
              <tbody>
                {pairRisks.map((p, i) => (
                  <tr key={i} className="border-b border-slate-700/20">
                    <td className="py-2 pr-4 font-mono text-slate-300">
                      {Array.isArray(p.vehicles) ? p.vehicles.join(' ↔ ') : `Pair ${i + 1}`}
                    </td>
                    <td className="py-2 pr-4 text-slate-200">
                      {p.ttc_seconds != null ? Number(p.ttc_seconds).toFixed(2) : '—'}
                    </td>
                    <td className="py-2 pr-4 text-slate-200">
                      {p.near_miss_probability != null ? Number(p.near_miss_probability).toFixed(4) : '—'}
                    </td>
                    <td className="py-2 text-slate-200">
                      {p.risk_score != null ? Number(p.risk_score).toFixed(4) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Comparison results if available */}
      {comparison.length > 0 && (
        <Card title="AI Method Comparison (Evaluation Results)">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700/50">
                  <th className="text-left py-2 pr-4">Controller</th>
                  <th className="text-left py-2 pr-4">Avg Wait (s)</th>
                  <th className="text-left py-2 pr-4">Avg Queue</th>
                  <th className="text-left py-2 pr-4">Throughput</th>
                  <th className="text-left py-2">Runs</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((row, i) => (
                  <tr key={i} className="border-b border-slate-700/20">
                    <td className="py-2 pr-4 font-medium text-slate-300">{row.controller}</td>
                    <td className="py-2 pr-4 text-slate-200">
                      {row.average_waiting_time_mean != null ? Number(row.average_waiting_time_mean).toFixed(2) : '—'}
                    </td>
                    <td className="py-2 pr-4 text-slate-200">
                      {row.average_queue_length_mean != null ? Number(row.average_queue_length_mean).toFixed(2) : '—'}
                    </td>
                    <td className="py-2 pr-4 text-slate-200">
                      {row.throughput_mean != null ? Number(row.throughput_mean).toFixed(0) : '—'}
                    </td>
                    <td className="py-2 text-slate-400">{row.successful_runs ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
