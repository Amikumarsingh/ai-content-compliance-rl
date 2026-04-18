import React, { useEffect, useState } from 'react'
import { PieChart, Pie, Cell, Tooltip, Legend, BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from 'recharts'
import { TrendingUp, Target, Clock, Award, BarChart2 } from 'lucide-react'
import { api } from '../hooks/useApi'

const VIOLATION_COLORS = ['#8b5cf6','#f87171','#fbbf24','#34d399','#60a5fa','#f472b6','#a78bfa','#2dd4bf','#fb923c']

const TOOLTIP = {
  contentStyle: {
    background: '#111827',
    border: '1px solid #1c2840',
    borderRadius: 10,
    fontSize: 12,
    color: '#e2e8f0',
    boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
  },
  cursor: { fill: 'rgba(255,255,255,0.03)' },
}

function KpiCard({ icon: Icon, label, value, accent = '#8b5cf6' }) {
  return (
    <div
      className="card p-5 transition-all duration-200 hover:scale-[1.01]"
      style={{ boxShadow: `0 1px 3px rgb(0 0 0 / 0.5), 0 0 0 1px rgba(255,255,255,0.03)` }}
    >
      <div className="flex items-center justify-between mb-3">
        <p className="label">{label}</p>
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center"
          style={{ background: `${accent}18`, border: `1px solid ${accent}25` }}
        >
          <Icon size={13} style={{ color: accent }} />
        </div>
      </div>
      <p className="text-2xl font-bold tabular-nums" style={{ color: 'var(--text-1)' }}>{value ?? '—'}</p>
    </div>
  )
}

export default function Analytics() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/analytics').then(r => setData(r.data)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="p-8 flex items-center gap-3 text-sm" style={{ color: 'var(--text-3)' }}>
      <div className="w-4 h-4 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
      Loading analytics…
    </div>
  )

  if (!data?.total) return (
    <div className="p-8 space-y-6">
      <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>Analytics</h1>
      <div className="card p-14 text-center">
        <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4"
          style={{ background: 'var(--accent-dim)', border: '1px solid rgba(139,92,246,0.2)' }}>
          <BarChart2 size={22} style={{ color: 'var(--accent)' }} />
        </div>
        <p className="text-sm font-medium" style={{ color: 'var(--text-2)' }}>No data yet</p>
        <p className="text-xs mt-1" style={{ color: 'var(--text-3)' }}>Moderate some content in the Playground first</p>
      </div>
    </div>
  )

  const decisionData = [
    { name: 'Approved', value: data.approved, fill: '#34d399' },
    { name: 'Rejected', value: data.rejected, fill: '#f87171' },
    { name: 'Edited',   value: data.edited,   fill: '#fbbf24' },
  ].filter(d => d.value > 0)

  const violationData = Object.entries(data.violation_breakdown ?? {})
    .map(([name, value]) => ({ name: name.replace(/_/g, ' '), value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8)

  return (
    <div className="p-8 space-y-8 max-w-5xl">
      <div>
        <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>Analytics</h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-3)' }}>Performance metrics across all moderation decisions</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard icon={Target}     label="Total"       value={data.total}                          accent="#8b5cf6" />
        <KpiCard icon={Award}      label="Avg Score"   value={data.avg_score?.toFixed(3)}          accent="#34d399" />
        <KpiCard icon={TrendingUp} label="Avg Reward"  value={data.avg_reward?.toFixed(3)}         accent="#fbbf24" />
        <KpiCard icon={Clock}      label="Avg Latency" value={`${data.avg_latency_ms?.toFixed(0)}ms`} accent="#60a5fa" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Donut */}
        <div className="card p-5 space-y-4">
          <p className="label">Decision Breakdown</p>
          <ResponsiveContainer width="100%" height={230}>
            <PieChart>
              <Pie data={decisionData} dataKey="value" nameKey="name"
                cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={4}>
                {decisionData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} stroke="transparent"
                    style={{ filter: `drop-shadow(0 0 6px ${entry.fill}60)` }} />
                ))}
              </Pie>
              <Tooltip {...TOOLTIP} />
              <Legend iconType="circle" iconSize={8}
                formatter={v => <span style={{ color: '#94a3b8', fontSize: 12 }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bar */}
        <div className="card p-5 space-y-4">
          <p className="label">Top Violations</p>
          {violationData.length === 0 ? (
            <div className="flex items-center justify-center h-[230px] text-sm" style={{ color: 'var(--text-3)' }}>
              No violations detected yet
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={violationData} layout="vertical" margin={{ left: 0, right: 20 }}>
                <XAxis type="number" tick={{ fill: '#475569', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={115} axisLine={false} tickLine={false} />
                <Tooltip {...TOOLTIP} />
                <Bar dataKey="value" radius={[0, 5, 5, 0]} maxBarSize={16}>
                  {violationData.map((_, i) => (
                    <Cell key={i} fill={VIOLATION_COLORS[i % VIOLATION_COLORS.length]}
                      style={{ filter: `drop-shadow(0 0 4px ${VIOLATION_COLORS[i % VIOLATION_COLORS.length]}50)` }} />
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
            <p className="label">Human Feedback Accuracy</p>
            <p className="text-3xl font-bold mt-1" style={{ color: '#34d399' }}>
              {(data.accuracy * 100).toFixed(1)}%
            </p>
          </div>
          <p className="text-xs text-right" style={{ color: 'var(--text-3)' }}>
            Based on thumbs up/down<br />feedback from the dashboard
          </p>
        </div>
      )}
    </div>
  )
}
