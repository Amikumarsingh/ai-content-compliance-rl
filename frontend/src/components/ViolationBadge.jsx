import React from 'react'

const STYLES = {
  hate_speech:    { bg: 'rgba(239,68,68,0.1)',   color: '#fca5a5', border: 'rgba(239,68,68,0.25)'   },
  violence:       { bg: 'rgba(249,115,22,0.1)',  color: '#fdba74', border: 'rgba(249,115,22,0.25)'  },
  harassment:     { bg: 'rgba(234,179,8,0.1)',   color: '#fde047', border: 'rgba(234,179,8,0.25)'   },
  adult_content:  { bg: 'rgba(236,72,153,0.1)',  color: '#f9a8d4', border: 'rgba(236,72,153,0.25)'  },
  misinformation: { bg: 'rgba(139,92,246,0.12)', color: '#c4b5fd', border: 'rgba(139,92,246,0.3)'   },
  spam:           { bg: 'rgba(59,130,246,0.1)',  color: '#93c5fd', border: 'rgba(59,130,246,0.25)'  },
  suspicious_link:{ bg: 'rgba(6,182,212,0.1)',   color: '#67e8f9', border: 'rgba(6,182,212,0.25)'   },
  engagement_bait:{ bg: 'rgba(20,184,166,0.1)',  color: '#5eead4', border: 'rgba(20,184,166,0.25)'  },
  illegal_content:{ bg: 'rgba(244,63,94,0.1)',   color: '#fda4af', border: 'rgba(244,63,94,0.25)'   },
}

const DEFAULT = { bg: 'rgba(100,116,139,0.1)', color: '#94a3b8', border: 'rgba(100,116,139,0.25)' }

export default function ViolationBadge({ type }) {
  const s = STYLES[type] ?? DEFAULT
  return (
    <span
      className="badge"
      style={{ background: s.bg, color: s.color, borderColor: s.border }}
    >
      {type.replace(/_/g, ' ')}
    </span>
  )
}
