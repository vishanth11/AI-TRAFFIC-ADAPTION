// pages/Metrics.jsx
import { useLiveData } from '../context/LiveDataContext'
import { Card, MetricCard, EmptyState } from '../components/ui'
import { Clock, Car, Activity, Fuel } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'

const METRIC_ROWS = [
  { key: 'average_waiting_time', label: 'Avg Waiting Time', unit: 's' },
  { key: 'total_waiting_time', label: 'Total Waiting Time', unit: 's' },
  { key: 'maximum_waiting_time', label: 'Max Waiting Time', unit: 's' },
  { key: 'average_queue_length', label: 'Avg Queue Length', unit: '' },
  { key: 'maximum_queue_length', label: 'Max Queue Length', unit: '' },
  { key: 'throughput', label: 'Throughput (vehicles)', unit: '' },
  { key: 'average_travel_time', label: 'Avg Travel Time', unit: 's' },
  { key: 'total_travel_time', label: 'Total Travel Time', unit: 's' },
  { key: 'number_of_stops', label: 'Number of Stops', unit: '' },
  { key: 'fuel_consumption_mg', label: 'Fuel Consumption', unit: 'mg' },
  { key: 'co2_emissions_mg', label: 'CO₂ Emissions', unit: 'mg' },
]

export default function Metrics() {
  const { metrics } = useLiveData()

  const current = metrics?.current ?? {}
  const waitHistory = metrics?.waiting_time_history ?? []
  const queueHistory = metrics?.queue_history ?? []
  const throughputHistory = metrics?.throughput_history ?? []

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Performance Metrics</h2>
        <p className="text-sm text-slate-400">Live SUMO traffic performance indicators</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Avg Waiting Time"
          value={current.average_waiting_time != null ? `${Number(current.average_waiting_time).toFixed(2)}s` : '—'}
          icon={Clock}
          accent="yellow"
        />
        <MetricCard
          label="Avg Queue"
          value={current.average_queue_length != null ? Number(current.average_queue_length).toFixed(2) : '—'}
          icon={Car}
          accent="blue"
        />
        <MetricCard
          label="Throughput"
          value={current.throughput != null ? String(current.throughput) : '—'}
          icon={Activity}
          accent="green"
        />
        <MetricCard
          label="CO₂ (mg)"
          value={current.co2_emissions_mg != null ? Number(current.co2_emissions_mg).toFixed(0) : '—'}
          icon={Fuel}
          accent="red"
        />
      </div>

      {/* Charts */}
      <MetricChart title="Average Waiting Time (s)" data={waitHistory} color="#facc15" />
      <MetricChart title="Average Queue Length" data={queueHistory} color="#60a5fa" />
      <MetricChart title="Throughput (vehicles arrived)" data={throughputHistory} color="#34d399" />

      {/* Full metrics table */}
      <Card title="All Metrics">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-500 border-b border-slate-700/50">
                <th className="text-left py-2 pr-4">Metric</th>
                <th className="text-right py-2">Value</th>
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map(({ key, label, unit }) => {
                const val = current[key]
                return (
                  <tr key={key} className="border-b border-slate-700/20">
                    <td className="py-2 pr-4 text-slate-400">{label}</td>
                    <td className="py-2 text-right text-slate-200 font-mono">
                      {val != null
                        ? `${Number(val).toFixed(typeof val === 'number' && val % 1 !== 0 ? 3 : 0)}${unit ? ' ' + unit : ''}`
                        : <span className="text-slate-600">—</span>
                      }
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function MetricChart({ title, data, color }) {
  return (
    <Card title={title}>
      {!data.length ? (
        <EmptyState message="No data yet — simulation must be running" />
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="time"
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickFormatter={v => `${Number(v).toFixed(0)}s`}
            />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} width={40} />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6 }}
              labelStyle={{ color: '#94a3b8' }}
              itemStyle={{ color }}
              labelFormatter={v => `t=${Number(v).toFixed(1)}s`}
            />
            <Line type="monotone" dataKey="value" stroke={color} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </Card>
  )
}
