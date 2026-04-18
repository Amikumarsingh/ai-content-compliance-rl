import React, { useEffect, useState } from 'react'
import { Shield, CheckCircle, XCircle, Edit3, Zap } from 'lucide-react'
import ContentCard from '../components/ContentCard'
import { api } from '../hooks/useApi'

function StatCard({ icon: Icon, label, value, color }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 flex items-center gap-4">
      <div className={`p-2.5 rounded-lg ${color}`}>
        <Icon size={20} />
      </div>
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
        <p className="text-2xl font-bold text-gray-100">{value ?? '—'}</p>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [logs, setLogs]  = useState([])

  async function load() {
    try {
      const r = await api.get('/analytics')
      setData(r.data)
      setLogs(r.data.recent || [])
    } catch {}
  }

  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t) }, [])

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-gray-400 text-sm mt-1">Live moderation feed · auto-refreshes every 10s</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Shield}       label="Total Moderated" value={data?.total}    color="bg-brand-500/20 text-brand-500" />
        <StatCard icon={CheckCircle}  label="Approved"        value={data?.approved} color="bg-green-500/20 text-green-400" />
        <StatCard icon={XCircle}      label="Rejected"        value={data?.rejected} color="bg-red-500/20 text-red-400"     />
        <StatCard icon={Zap}          label="Avg Reward"      value={data?.avg_reward?.toFixed(3)} color="bg-purple-500/20 text-purple-400" />
      </div>

      <div>
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Recent Decisions</h2>
        {logs.length === 0 ? (
          <div className="text-center py-16 text-gray-600">
            <Shield size={40} className="mx-auto mb-3 opacity-30" />
            <p>No moderation logs yet. Use the Playground to test content.</p>
          </div>
        ) : (
          <div className="space-y-3">
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
