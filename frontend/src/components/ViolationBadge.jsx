import React from 'react'

const COLORS = {
  hate_speech:    'bg-red-500/20 text-red-400 border-red-500/30',
  violence:       'bg-orange-500/20 text-orange-400 border-orange-500/30',
  harassment:     'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  adult_content:  'bg-pink-500/20 text-pink-400 border-pink-500/30',
  misinformation: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  spam:           'bg-blue-500/20 text-blue-400 border-blue-500/30',
  suspicious_link:'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  engagement_bait:'bg-teal-500/20 text-teal-400 border-teal-500/30',
  illegal_content:'bg-rose-500/20 text-rose-400 border-rose-500/30',
}

export default function ViolationBadge({ type }) {
  const cls = COLORS[type] || 'bg-gray-500/20 text-gray-400 border-gray-500/30'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${cls}`}>
      {type.replace(/_/g, ' ')}
    </span>
  )
}
