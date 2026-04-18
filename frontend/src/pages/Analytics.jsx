import React, { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import { TrendingUp, Target, Clock, Award } from 'lucide-react'
import { api } from '../hooks/useApi'

const VIOLATION_COLORS = [
  '#6366f1','#ef4444','#f59e0b','#8b5cf6',
  '#10b981','#06b6d4','#f97316','#ec4899','#84cc16',
]

const TOOLTIP_STYLE = {
  background: '#161b27',
  border: '1px solid #1e2535',
  borderRadius: 8,
  fontSize: 12,
  color: '#e2e8f0',
}

function KpiCard({ icon: Icon, label, value, sub }) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="section-title">{label}</p>
        <Icon size={14} className="text-slate-600" />
      </div>
      <p className="text-2xl font-bold tabular-nums">{value ?? '—'}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  )
}

export default function Analytics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/analytics')
      .then(r => setData(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="p-8 flex items-center gap-2 text-slate-500 text-sm">
      <div className="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
      Loading analytics…
    </div>
  )

  if (!data?.total) return (
    <div className="p-8 space-y-4">
      <h1 className="text-xl font-bold">Analytics</h1>
      <div className="card p-12 text-center">
        <TrendingUp size={32} className="mx-auto mb-3 text-slate-700" />
        <p className="text-sm text-slate-500">No data yet.</p>
        <p className="text-xs text-slate-600 mt-1">Moderate some content in the Playground first.</p>
      </div>
    </div>
  )

  const decisionData = [
    { name: 'Approved', value: data.approved, fill: '#10b981' },
    { name: 'Rejected', value: data.rejected, fill: '#ef4444' },
    { name: 'Edited',   value: data.edited,   fill: '#f59e0b' },
  ].filter(d => d.value > 0)

  const violationData = Object.entries(data.violation_breakdown ?? {})
    .map(([name, value]) => ({ name: name.replace(/_/g, ' '), value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8)

  return (
    <div className="p-8 space-y-8 max-w-5xl">
      <div>
        <h1 className="text-xl font-bold text-slate-100">Analytics</h1>
        <p className="text-sm text-slate-500 mt-0.5">Performance metrics across all moderation decisions</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard icon={Target}    label="Total Moderated" value={data.total} />
        <KpiCard icon={Award}     label="Avg Score"       value={data.avg_score?.toFixed(3)}  sub="compliance score" />
        <KpiCard icon={TrendingUp}label="Avg Reward"      value={data.avg_reward?.toFixed(3)} sub="RL reward signal" />
        <KpiCard icon={Clock}     label="Avg Latency"     value={`${data.avg_latency_ms?.toFixed(0)}ms`} sub="per request" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Decision pie */}
        <div className="card p-5 space-y-4">
          <p className="section-title">Decision Breakdown</p>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={decisionData}
                dataKey="value"
                nameKey="name"
                cx="50%" cy="50%"
                innerRadius={55}
                outerRadius={85}
                paddingAngle={3}
              >
                {decisionData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} stroke="transparent" />
                ))}
              </Pie>
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend
                iconType="circle"
                iconSize={8}
                formatter={v => <span style={{ color: '#94a3b8', fontSize: 12 }}>{v}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Violations bar */}
        <div className="card p-5 space-y-4">
          <p className="section-title">Top Violations</p>
          {violationData.length === 0 ? (
            <div className="flex items-center justify-center h-[220px] text-slate-600 text-sm">
              No violations detected yet
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={violationData} layout="vertical" margin={{ left: 0, right: 16 }}>
                <XAxis type="number" tick={{ fill: '#475569', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={115} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18}>
                  {violationData.map((_, i) => (
                    <Cell key={i} fill={VIOLATION_COLORS[i % VIOLATION_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Accuracy */}
      {data.accuracy != null && (
        <div className="card p-5 flex items-center justify-between">
          <div>
            <p className="section-title">Human Feedback Accuracy</p>
            <p className="text-2xl font-bold mt-1">{(data.accuracy * 100).toFixed(1)}%</p>
          </div>
          <div className="text-xs text-slate-500 text-right">
            Based on thumbs up/down<br />feedback from the dashboard
          </div>
        </div>
      )}
    </div>
  )
}
