// components/Sidebar.jsx
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Radio, Brain, ShieldAlert, Eye,
  ScrollText, BarChart3, Activity, Settings, ChevronLeft, ChevronRight
} from 'lucide-react'

const NAV = [
  { label: 'Main', items: [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/junctions', icon: Radio, label: 'Live Junctions' },
    { to: '/decisions', icon: Brain, label: 'AI Decisions' },
    { to: '/safety', icon: ShieldAlert, label: 'Safety & Emergency' },
    { to: '/perception', icon: Eye, label: 'Perception & Risk' },
    { to: '/logs', icon: ScrollText, label: 'System Logs' },
    { to: '/metrics', icon: BarChart3, label: 'Performance Metrics' },
  ]},
  { label: 'System', items: [
    { to: '/system', icon: Activity, label: 'System Status' },
    { to: '/config', icon: Settings, label: 'Configuration' },
  ]},
]

export default function Sidebar({ collapsed, onToggle }) {
  return (
    <aside className={`
      fixed left-0 top-0 h-full z-30 flex flex-col
      bg-slate-900 border-r border-slate-700/50
      transition-all duration-200
      ${collapsed ? 'w-14' : 'w-56'}
    `}>
      {/* Logo */}
      <div className="flex items-center gap-2 px-3 py-4 border-b border-slate-700/50 min-h-[60px]">
        <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0">
          <Radio className="w-4 h-4 text-white" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <p className="text-xs font-bold text-white leading-tight">AI TRAFFIC</p>
            <p className="text-xs text-slate-400 leading-tight">Admin</p>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-4">
        {NAV.map(section => (
          <div key={section.label}>
            {!collapsed && (
              <p className="text-xs text-slate-500 uppercase tracking-wider px-2 mb-1">
                {section.label}
              </p>
            )}
            <ul className="space-y-0.5">
              {section.items.map(({ to, icon: Icon, label }) => (
                <li key={to}>
                  <NavLink
                    to={to}
                    end={to === '/'}
                    className={({ isActive }) => `
                      flex items-center gap-2.5 px-2 py-2 rounded-lg text-sm transition-colors
                      ${isActive
                        ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}
                    `}
                    title={collapsed ? label : undefined}
                  >
                    <Icon className="w-4 h-4 flex-shrink-0" />
                    {!collapsed && <span className="truncate">{label}</span>}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        className="flex items-center justify-center p-3 border-t border-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  )
}
