import React, { useEffect, useState } from 'react'
import { Shield, CheckCircle2, XCircle, Zap, RefreshCw, TrendingUp } from 'lucide-react'
import ContentCard from '../components/ContentCard'
import { api } from '../hooks/useApi'

const STATS = [
  { key: 'total',      label: 'Total Moderated', icon: Shield,       accent: '#8b5cf6', glow: 'rgba(139,92,246,0.15)' },
  { key: 'approved',   label: 'Approved',         icon: CheckCircle2, accent: '#34d399', glow: 'rgba(52,211,153,0.15)'  },
  { key: 'rejected',   label: 'Rejected',          icon: XCircle,      accent: '#f87171', glow: 'rgba(248,113,113,0.15)' },
  { key: 'avg_reward', label: 'Avg Reward',        icon: Zap,          accent: '#fbbf24', glow: 'rgba(251,191,36,0.15)',  fmt: v => v?.toFixed(3) },
]

function StatCard({ label, value, icon: Icon, accent, glow }) {
  return (
    <div
      className="card p-5 flex items-center gap-4 transition-all duration-200 hover:scale-[1.01]"
      style={{ boxShadow: `0 1px 3px rgb(0 0 0 / 0.5), 0 0 0 1px rgba(255,255,255,0.03)` }}
    >
      <div
        className="flex items-center justify-center w-10 h-10 rounded-xl flex-shrink-0"
        style={{ background: `${accent}18`, border: `1px solid ${accent}30`, boxShadow: `0 0 16px ${glow}` }}
      >
        <Icon size={18} style={{ color: accent }} />
      </div>
      <div>
        <p className="label">{label}</p>
        <p className="text-2xl font-bold mt-0.5 tabular-nums" style={{ color: 'var(--text-1)' }}>
          {value ?? '—'}
        </p>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [data,       setData]       = useState(null)
  const [logs,       setLogs]       = useState([])
  const [refreshing, setRefreshing] = useState(false)

  async function load(spinner = false) {
    if (spinner) setRefreshing(true)
    try {
      const r = await api.get('/analytics')
      setData(r.data)
      setLogs(r.data.recent ?? [])
    } catch {}
    finally { setRefreshing(false) }
  }

  useEffect(() => {
    load()
    const t = setInterval(load, 10_000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="p-8 space-y-8 max-w-5xl">

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>Dashboard</h1>
          <p className="text-sm mt-0.5" style={{ color: 'var(--text-3)' }}>
            Live moderation feed · auto-refreshes every 10s
          </p>
        </div>
        <button onClick={() => load(true)} className="btn-secondary">
          <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {STATS.map(({ key, label, icon, accent, glow, fmt }) => (
          <StatCard
            key={key}
            label={label}
            value={fmt ? fmt(data?.[key]) : data?.[key]}
            icon={icon}
            accent={accent}
            glow={glow}
          />
        ))}
      </div>

      {/* Feed */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <p className="label">Recent Decisions</p>
          {logs.length > 0 && (
            <span className="text-xs" style={{ color: 'var(--text-3)' }}>{logs.length} entries</span>
          )}
        </div>

        {logs.length === 0 ? (
          <div
            className="card p-14 text-center"
            style={{ background: 'linear-gradient(135deg, var(--bg-card), var(--bg-elevated))' }}
          >
            <div
              className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4"
              style={{ background: 'var(--accent-dim)', border: '1px solid rgba(139,92,246,0.2)' }}
            >
              <Shield size={22} style={{ color: 'var(--accent)' }} />
            </div>
            <p className="text-sm font-medium" style={{ color: 'var(--text-2)' }}>No moderation logs yet</p>
            <p className="text-xs mt-1" style={{ color: 'var(--text-3)' }}>
              Use the Playground to test content
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {logs.map(log => (
              <ContentCard
                key={log.id}
                log={log}
                onFeedback={(id, correct) =>
                  setLogs(prev => prev.map(l => l.id === id ? { ...l, correct } : l))
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
