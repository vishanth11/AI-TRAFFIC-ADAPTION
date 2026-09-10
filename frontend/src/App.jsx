import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { LiveDataProvider } from './context/LiveDataContext'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import Dashboard from './pages/Dashboard'
import Junctions from './pages/Junctions'
import Decisions from './pages/Decisions'
import Safety from './pages/Safety'
import Perception from './pages/Perception'
import Logs from './pages/Logs'
import Metrics from './pages/Metrics'
import SystemStatus from './pages/SystemStatus'
import Config from './pages/Config'

export default function App() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <BrowserRouter>
      <LiveDataProvider>
        <div className="min-h-screen bg-slate-950 text-white">
          <TopBar />
          <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
          <main
            className="pt-[60px] transition-all duration-200"
            style={{ marginLeft: collapsed ? 56 : 224 }}
          >
            <div className="p-6">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/junctions" element={<Junctions />} />
                <Route path="/decisions" element={<Decisions />} />
                <Route path="/safety" element={<Safety />} />
                <Route path="/perception" element={<Perception />} />
                <Route path="/logs" element={<Logs />} />
                <Route path="/metrics" element={<Metrics />} />
                <Route path="/system" element={<SystemStatus />} />
                <Route path="/config" element={<Config />} />
              </Routes>
            </div>
          </main>
        </div>
      </LiveDataProvider>
    </BrowserRouter>
  )
}
