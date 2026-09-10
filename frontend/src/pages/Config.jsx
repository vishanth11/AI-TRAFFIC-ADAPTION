// pages/Config.jsx
import { Card } from '../components/ui'
import { Settings } from 'lucide-react'

const COMMANDS = [
  { label: 'Normal run (200 steps)', cmd: 'python run_system.py --steps 200 --seed 1' },
  { label: 'Emergency demo', cmd: 'python run_system.py --demo --steps 200 --seed 1' },
  { label: 'Start dashboard API', cmd: 'uvicorn dashboard_api:app --host 0.0.0.0 --port 8000 --reload' },
  { label: 'Dashboard + controller', cmd: 'python dashboard_runner.py --steps 200' },
  { label: 'Phase 2.1 evaluation', cmd: 'python src\\evaluation\\phase2_evaluation.py --seeds 1 2 3 --steps 200 --warmup 10' },
  { label: 'Run tests', cmd: 'python -m pytest -q' },
]

const ENDPOINTS = [
  { method: 'GET', path: '/api/status', desc: 'System status snapshot' },
  { method: 'GET', path: '/api/junctions', desc: 'All junction states' },
  { method: 'GET', path: '/api/junctions/{id}', desc: 'Single junction state' },
  { method: 'GET', path: '/api/decisions', desc: 'AI decision log' },
  { method: 'GET', path: '/api/safety', desc: 'Safety assessment' },
  { method: 'GET', path: '/api/emergencies', desc: 'Active emergency vehicles' },
  { method: 'GET', path: '/api/metrics', desc: 'Performance metrics + history' },
  { method: 'GET', path: '/api/logs', desc: 'Runtime log entries' },
  { method: 'GET', path: '/api/comparison', desc: 'Evaluation comparison results' },
  { method: 'WS', path: '/ws/live', desc: 'Live WebSocket stream' },
]

export default function Config() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-white">Configuration</h2>
        <p className="text-sm text-slate-400">System commands and API reference</p>
      </div>

      <Card title="Runtime Commands">
        <div className="space-y-3">
          {COMMANDS.map(({ label, cmd }) => (
            <div key={cmd}>
              <p className="text-xs text-slate-500 mb-1">{label}</p>
              <code className="block text-xs bg-slate-900 text-emerald-300 rounded px-3 py-2 font-mono select-all">
                {cmd}
              </code>
            </div>
          ))}
        </div>
      </Card>

      <Card title="API Endpoints">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-slate-700/50">
                <th className="text-left py-2 pr-4">Method</th>
                <th className="text-left py-2 pr-4">Path</th>
                <th className="text-left py-2">Description</th>
              </tr>
            </thead>
            <tbody>
              {ENDPOINTS.map(({ method, path, desc }) => (
                <tr key={path} className="border-b border-slate-700/20">
                  <td className="py-2 pr-4">
                    <span className={`font-mono font-bold ${method === 'WS' ? 'text-purple-400' : 'text-blue-400'}`}>
                      {method}
                    </span>
                  </td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{path}</td>
                  <td className="py-2 text-slate-400">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Simulation Files">
        <div className="space-y-1 text-xs font-mono text-slate-400">
          <p><span className="text-slate-500">Normal:</span> simulation/configs/corridor.sumocfg</p>
          <p><span className="text-slate-500">Emergency:</span> simulation/configs/corridor_emergency.sumocfg</p>
          <p><span className="text-slate-500">Four-way:</span> simulation/configs/four_way.sumocfg</p>
        </div>
      </Card>
    </div>
  )
}
