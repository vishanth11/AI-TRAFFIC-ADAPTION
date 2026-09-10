// components/ui.jsx — shared reusable UI primitives

export function StatusBadge({ status }) {
  const map = {
    SAFE: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    LOW: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    MEDIUM: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    HIGH: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    CRITICAL: 'bg-red-500/20 text-red-400 border-red-500/30',
    CONNECTED: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    DISCONNECTED: 'bg-red-500/20 text-red-400 border-red-500/30',
    CONNECTING: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    ACTIVE: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    INACTIVE: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    EXECUTED: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    REJECTED: 'bg-red-500/20 text-red-400 border-red-500/30',
    UNAVAILABLE: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    APPROVED: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    running: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    stopped: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    error: 'bg-red-500/20 text-red-400 border-red-500/30',
  }
  const cls = map[status] ?? 'bg-slate-500/20 text-slate-400 border-slate-500/30'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${cls}`}>
      {status}
    </span>
  )
}

export function MetricCard({ label, value, sub, icon: Icon, accent = 'blue' }) {
  const accents = {
    blue: 'border-blue-500/30 bg-blue-500/5',
    green: 'border-emerald-500/30 bg-emerald-500/5',
    yellow: 'border-yellow-500/30 bg-yellow-500/5',
    red: 'border-red-500/30 bg-red-500/5',
    purple: 'border-purple-500/30 bg-purple-500/5',
  }
  return (
    <div className={`rounded-xl border p-4 ${accents[accent] ?? accents.blue}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">{label}</p>
          <p className="text-2xl font-bold text-white">{value ?? '—'}</p>
          {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
        </div>
        {Icon && <Icon className="w-5 h-5 text-slate-500 mt-1" />}
      </div>
    </div>
  )
}

export function Card({ title, children, className = '' }) {
  return (
    <div className={`rounded-xl border border-slate-700/50 bg-slate-800/50 ${className}`}>
      {title && (
        <div className="px-4 py-3 border-b border-slate-700/50">
          <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  )
}

export function LoadingState({ message = 'Loading...' }) {
  return (
    <div className="flex items-center justify-center h-40 text-slate-400 text-sm">
      <div className="flex items-center gap-2">
        <div className="w-4 h-4 border-2 border-slate-600 border-t-blue-400 rounded-full animate-spin" />
        {message}
      </div>
    </div>
  )
}

export function EmptyState({ message = 'No data available' }) {
  return (
    <div className="flex items-center justify-center h-32 text-slate-500 text-sm">
      {message}
    </div>
  )
}

export function ErrorState({ message = 'Backend unavailable. Trying to reconnect...' }) {
  return (
    <div className="flex items-center justify-center h-32 text-red-400 text-sm">
      <div className="text-center">
        <p className="font-medium">Connection Error</p>
        <p className="text-slate-500 mt-1">{message}</p>
      </div>
    </div>
  )
}

export function CheckRow({ label, value }) {
  const ok = value === true
  const na = value === undefined || value === null
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-700/30 last:border-0">
      <span className="text-sm text-slate-300">{label}</span>
      {na ? (
        <span className="text-xs text-slate-500">—</span>
      ) : ok ? (
        <span className="text-xs text-emerald-400 font-medium">✓ PASS</span>
      ) : (
        <span className="text-xs text-red-400 font-medium">✗ FAIL</span>
      )}
    </div>
  )
}

export function DataSourceTag({ source }) {
  return (
    <span className="text-xs text-slate-500 bg-slate-700/50 px-2 py-0.5 rounded">
      Source: {source}
    </span>
  )
}
