import React, { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { api } from '../hooks/useApi'

const PIE_COLORS = ['#4f6ef7','#ef4444','#f59e0b','#8b5cf6','#10b981','#06b6d4','#f97316','#ec4899','#6366f1']

export default function Analytics() {
  const [data, setData] = useState(null)

  useEffect(() => {
    api.get('/analytics').then(r => setData(r.data)).catch(() => {})
  }, [])

  if (!data) return <div className="p-8 text-gray-500">Loading analytics…</div>
  if (!data.total) return (
    <div className="p-8 text-gray-500">No data yet. Moderate some content first.</div>
  )

  const decisionData = [
    { name: 'Approved', value: data.approved, fill: '#10b981' },
    { name: 'Rejected', value: data.rejected, fill: '#ef4444' },
    { name: 'Edited',   value: data.edited,   fill: '#f59e0b' },
  ]

  const violationData = Object.entries(data.violation_breakdown || {})
    .map(([name, value]) => ({ name: name.replace(/_/g, ' '), value }))
    .sort((a, b) => b.value - a.value)

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Analytics</h1>
        <p className="text-gray-400 text-sm mt-1">Performance metrics across all moderation decisions</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Total',        value: data.total },
          { label: 'Avg Score',    value: data.avg_score?.toFixed(3) },
          { label: 'Avg Reward',   value: data.avg_reward?.toFixed(3) },
          { label: 'Accuracy',     value: data.accuracy != null ? `${(data.accuracy * 100).toFixed(1)}%` : 'N/A' },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
            <p className="text-2xl font-bold mt-1">{value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Decision breakdown */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Decision Breakdown</h2>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={decisionData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`}>
                {decisionData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Violation types */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Top Violations</h2>
          {violationData.length === 0 ? (
            <p className="text-gray-600 text-sm">No violations detected yet.</p>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={violationData} layout="vertical">
                <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} width={110} />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {violationData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  )
}
