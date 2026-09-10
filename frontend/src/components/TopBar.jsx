// components/TopBar.jsx
import { useLiveData } from '../context/LiveDataContext'

function formatTime(seconds) {
  if (seconds == null) return '00:00'
  const m = Math.floor(seconds / 60).toString().padStart(2, '0')
  const s = Math.floor(seconds % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

export default function TopBar() {
  const { wsStatus, status } = useLiveData()

  const sysStatus = status?.system_status ?? 'stopped'
  const simTime = status?.simulation_time ?? 0

  const wsLabel = wsStatus === 'connected' ? 'Connected' : wsStatus === 'connecting' ? 'Connecting' : 'Disconnected'
  const wsColor = wsStatus === 'connected' ? 'text-emerald-400' : wsStatus === 'connecting' ? 'text-yellow-400' : 'text-red-400'
  const dotColor = wsStatus === 'connected' ? 'bg-emerald-400' : wsStatus === 'connecting' ? 'bg-yellow-400' : 'bg-red-400'

  return (
    <header className="fixed top-0 right-0 left-0 z-20 h-[60px] bg-slate-900/95 border-b border-slate-700/50 backdrop-blur flex items-center px-4 gap-4">
      <div className="flex-1">
        <h1 className="text-sm font-bold text-white tracking-wide">AI TRAFFIC ADAPTATION</h1>
        <p className="text-xs text-slate-400">Admin Control Center</p>
      </div>

      <div className="flex items-center gap-4 text-xs">
        {/* WS status */}
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${dotColor} ${wsStatus === 'connecting' ? 'animate-pulse' : ''}`} />
          <span className={wsColor}>{wsLabel}</span>
        </div>

        {/* SUMO status */}
        <div className="text-slate-400">
          SUMO: <span className={sysStatus === 'running' ? 'text-emerald-400' : 'text-slate-500'}>
            {sysStatus === 'running' ? 'Running' : sysStatus === 'stopped' ? 'Stopped' : sysStatus}
          </span>
        </div>

        {/* Sim time */}
        <div className="text-slate-400">
          Sim Time: <span className="text-white font-mono">{formatTime(simTime)}</span>
        </div>
      </div>
    </header>
  )
}
