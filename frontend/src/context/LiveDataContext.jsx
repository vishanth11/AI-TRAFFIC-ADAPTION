// context/LiveDataContext.jsx
import { createContext, useContext, useEffect, useReducer, useRef } from 'react'
import { createWebSocket, fetchStatus, fetchJunctions, fetchDecisions,
         fetchSafety, fetchEmergencies, fetchMetrics, fetchLogs } from '../services/api'

const LiveDataContext = createContext(null)

const initialState = {
  wsStatus: 'connecting',   // connecting | connected | disconnected
  status: null,
  junctions: [],
  decisions: [],
  safety: null,
  emergencies: [],
  metrics: null,
  logs: [],
  latestDecision: null,
}

function reducer(state, action) {
  switch (action.type) {
    case 'WS_OPEN':
      return { ...state, wsStatus: 'connected' }
    case 'WS_CLOSE':
      return { ...state, wsStatus: 'disconnected' }
    case 'INITIAL_STATE':
      return {
        ...state,
        status: action.payload.status,
        junctions: action.payload.junctions ?? state.junctions,
        decisions: action.payload.decisions ?? state.decisions,
        safety: action.payload.safety ?? state.safety,
        emergencies: action.payload.emergencies ?? state.emergencies,
        metrics: action.payload.metrics ?? state.metrics,
        logs: action.payload.logs ?? state.logs,
        latestDecision: action.payload.decisions?.[0] ?? state.latestDecision,
      }
    case 'LIVE_UPDATE':
      return {
        ...state,
        status: action.payload.status ?? state.status,
        junctions: action.payload.junctions ?? state.junctions,
        safety: action.payload.safety ?? state.safety,
        emergencies: action.payload.emergencies ?? state.emergencies,
        logs: action.payload.logs ?? state.logs,
        latestDecision: action.payload.latest_decision?.[0] ?? state.latestDecision,
      }
    case 'REST_STATUS':
      return { ...state, status: action.payload }
    case 'REST_JUNCTIONS':
      return { ...state, junctions: action.payload }
    case 'REST_DECISIONS':
      return { ...state, decisions: action.payload }
    case 'REST_SAFETY':
      return { ...state, safety: action.payload }
    case 'REST_EMERGENCIES':
      return { ...state, emergencies: action.payload }
    case 'REST_METRICS':
      return { ...state, metrics: action.payload }
    case 'REST_LOGS':
      return { ...state, logs: action.payload }
    default:
      return state
  }
}

export function LiveDataProvider({ children }) {
  const [data, dispatch] = useReducer(reducer, initialState)
  const wsRef = useRef(null)

  // Load initial REST data
  useEffect(() => {
    async function loadInitial() {
      try {
        const [status, junctions, decisions, safety, emergencies, metrics, logs] =
          await Promise.allSettled([
            fetchStatus(), fetchJunctions(), fetchDecisions(),
            fetchSafety(), fetchEmergencies(), fetchMetrics(), fetchLogs(),
          ])
        if (status.status === 'fulfilled') dispatch({ type: 'REST_STATUS', payload: status.value })
        if (junctions.status === 'fulfilled') dispatch({ type: 'REST_JUNCTIONS', payload: junctions.value })
        if (decisions.status === 'fulfilled') dispatch({ type: 'REST_DECISIONS', payload: decisions.value })
        if (safety.status === 'fulfilled') dispatch({ type: 'REST_SAFETY', payload: safety.value })
        if (emergencies.status === 'fulfilled') dispatch({ type: 'REST_EMERGENCIES', payload: emergencies.value })
        if (metrics.status === 'fulfilled') dispatch({ type: 'REST_METRICS', payload: metrics.value })
        if (logs.status === 'fulfilled') dispatch({ type: 'REST_LOGS', payload: logs.value })
      } catch (_) {}
    }
    loadInitial()
  }, [])

  // WebSocket
  useEffect(() => {
    wsRef.current = createWebSocket(
      (msg) => {
        if (msg.type === 'initial_state') dispatch({ type: 'INITIAL_STATE', payload: msg })
        else if (msg.type === 'live_update') dispatch({ type: 'LIVE_UPDATE', payload: msg })
      },
      () => dispatch({ type: 'WS_OPEN' }),
      () => dispatch({ type: 'WS_CLOSE' }),
    )
    return () => wsRef.current?.close()
  }, [])

  return (
    <LiveDataContext.Provider value={data}>
      {children}
    </LiveDataContext.Provider>
  )
}

export function useLiveData() {
  return useContext(LiveDataContext)
}
