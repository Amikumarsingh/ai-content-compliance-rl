import React from 'react'

const STYLES = {
  hate_speech:    'bg-red-500/10 text-red-400 border-red-500/20',
  violence:       'bg-orange-500/10 text-orange-400 border-orange-500/20',
  harassment:     'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  adult_content:  'bg-pink-500/10 text-pink-400 border-pink-500/20',
  misinformation: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  spam:           'bg-blue-500/10 text-blue-400 border-blue-500/20',
  suspicious_link:'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  engagement_bait:'bg-teal-500/10 text-teal-400 border-teal-500/20',
  illegal_content:'bg-rose-500/10 text-rose-400 border-rose-500/20',
}

export default function ViolationBadge({ type }) {
  const cls = STYLES[type] ?? 'bg-slate-500/10 text-slate-400 border-slate-500/20'
  return (
    <span className={`badge ${cls}`}>
      {type.replace(/_/g, ' ')}
    </span>
  )
}
