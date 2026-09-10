// services/api.js — REST + WebSocket data layer

const BASE = '/api'
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`

export async function fetchStatus() {
  const r = await fetch(`${BASE}/status`)
  return r.json()
}

export async function fetchJunctions() {
  const r = await fetch(`${BASE}/junctions`)
  return r.json()
}

export async function fetchJunction(id) {
  const r = await fetch(`${BASE}/junctions/${id}`)
  if (!r.ok) return null
  return r.json()
}

export async function fetchDecisions() {
  const r = await fetch(`${BASE}/decisions`)
  return r.json()
}

export async function fetchSafety() {
  const r = await fetch(`${BASE}/safety`)
  return r.json()
}

export async function fetchEmergencies() {
  const r = await fetch(`${BASE}/emergencies`)
  return r.json()
}

export async function fetchMetrics() {
  const r = await fetch(`${BASE}/metrics`)
  return r.json()
}

export async function fetchLogs() {
  const r = await fetch(`${BASE}/logs`)
  return r.json()
}

export async function fetchComparison() {
  const r = await fetch(`${BASE}/comparison`)
  return r.json()
}

export function createWebSocket(onMessage, onOpen, onClose) {
  let ws
  let reconnectTimer

  function connect() {
    ws = new WebSocket(WS_URL)

    ws.onopen = () => {
      clearTimeout(reconnectTimer)
      onOpen?.()
    }

    ws.onmessage = (e) => {
      try {
        onMessage(JSON.parse(e.data))
      } catch (_) {}
    }

    ws.onclose = () => {
      onClose?.()
      reconnectTimer = setTimeout(connect, 3000)
    }

    ws.onerror = () => ws.close()
  }

  connect()

  return {
    close() {
      clearTimeout(reconnectTimer)
      ws?.close()
    },
    send(data) {
      if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data))
    },
  }
}
