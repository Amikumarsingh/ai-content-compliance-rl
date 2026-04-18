import React, { useEffect, useState } from 'react'
import { Shield, CheckCircle2, XCircle, Pencil, Zap, RefreshCw } from 'lucide-react'
import ContentCard from '../components/ContentCard'
import { api } from '../hooks/useApi'

function StatCard({ icon: Icon, label, value, iconClass }) {
  return (
    <div className="card p-5 flex items-center gap-4">
      <div className={`p-2.5 rounded-lg border ${iconClass}`}>
        <Icon size={18} />
      </div>
      <div>
        <p className="section-title">{label}</p>
        <p className="text-2xl font-bold text-slate-100 mt-0.5 tabular-nums">{value ?? '—'}</p>
      </div>
    </div>
  )
}

function PageHeader({ title, subtitle, onRefresh, refreshing }) {
  return (
    <div className="flex items-start justify-between">
      <div>
        <h1 className="text-xl font-bold text-slate-100">{title}</h1>
        <p className="text-sm text-slate-500 mt-0.5">{subtitle}</p>
      </div>
      {onRefresh && (
        <button onClick={onRefresh} className="btn-ghost">
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      )}
    </div>
  )
}

export default function Dashboard() {
  const [data,       setData]       = useState(null)
  const [logs,       setLogs]       = useState([])
  const [refreshing, setRefreshing] = useState(false)

  async function load(showSpinner = false) {
    if (showSpinner) setRefreshing(true)
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
      <PageHeader
        title="Dashboard"
        subtitle="Live moderation feed · auto-refreshes every 10s"
        onRefresh={() => load(true)}
        refreshing={refreshing}
      />

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Shield}      label="Total"      value={data?.total}                    iconClass="bg-brand-500/10 border-brand-500/20 text-brand-400" />
        <StatCard icon={CheckCircle2}label="Approved"   value={data?.approved}                 iconClass="bg-emerald-500/10 border-emerald-500/20 text-emerald-400" />
        <StatCard icon={XCircle}     label="Rejected"   value={data?.rejected}                 iconClass="bg-red-500/10 border-red-500/20 text-red-400" />
        <StatCard icon={Zap}         label="Avg Reward" value={data?.avg_reward?.toFixed(3)}   iconClass="bg-purple-500/10 border-purple-500/20 text-purple-400" />
      </div>

      {/* Feed */}
      <div className="space-y-3">
        <p className="section-title">Recent Decisions</p>
        {logs.length === 0 ? (
          <div className="card p-12 text-center">
            <Shield size={32} className="mx-auto mb-3 text-slate-700" />
            <p className="text-sm text-slate-500">No moderation logs yet.</p>
            <p className="text-xs text-slate-600 mt-1">Use the Playground to test content.</p>
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
