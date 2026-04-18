import React, { useState } from 'react'
import { Send, Loader2, CheckCircle2, XCircle, Pencil, Sparkles } from 'lucide-react'
import ViolationBadge from '../components/ViolationBadge'
import { api } from '../hooks/useApi'

const EXAMPLES = [
  { label: 'Clean welcome',     text: 'Hello! Welcome to our community. Looking forward to connecting!' },
  { label: 'Obvious spam',      text: 'BUY NOW!!! Limited offer!!! Free prizes! bit.ly/scam' },
  { label: 'Hate speech',       text: 'I hate those people. They should all die!' },
  { label: 'Subtle harassment', text: 'Some people are just too ignorant to understand these topics. Sad.' },
  { label: 'Health misinfo',    text: "This supplement cured my diabetes in 3 days! Doctors don't want you to know. bit.ly/cure" },
  { label: 'Sarcasm trap',      text: 'I will kill them with my cooking 😂 — my chili is dangerously good.' },
]

const DECISIONS = {
  approve: { icon: CheckCircle2, label: 'Approved', color: '#34d399', bg: 'rgba(52,211,153,0.07)',  border: 'rgba(52,211,153,0.2)',  glow: 'rgba(52,211,153,0.12)'  },
  reject:  { icon: XCircle,      label: 'Rejected', color: '#f87171', bg: 'rgba(248,113,113,0.07)', border: 'rgba(248,113,113,0.2)', glow: 'rgba(248,113,113,0.12)' },
  edit:    { icon: Pencil,        label: 'Edit',     color: '#fbbf24', bg: 'rgba(251,191,36,0.07)',  border: 'rgba(251,191,36,0.2)',  glow: 'rgba(251,191,36,0.12)'  },
}

function ScoreBar({ value }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.7 ? '#34d399' : value >= 0.4 ? '#fbbf24' : '#f87171'
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <span className="label">Compliance Score</span>
        <span className="text-sm font-mono font-semibold" style={{ color }}>{value.toFixed(2)}</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-elevated)' }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color, boxShadow: `0 0 8px ${color}60` }}
        />
      </div>
    </div>
  )
}

export default function Playground() {
  const [content, setContent] = useState('')
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  async function moderate() {
    if (!content.trim()) return
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await api.post('/moderate', { content })
      setResult(r.data)
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Moderation failed. Check your API key in the API Keys page.')
    } finally { setLoading(false) }
  }

  const d = result ? DECISIONS[result.decision] : null

  return (
    <div className="p-8 max-w-3xl space-y-6">

      {/* Header */}
      <div>
        <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>Playground</h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-3)' }}>
          Test content through the full 4-step RL moderation pipeline
        </p>
      </div>

      {/* Examples */}
      <div className="space-y-2">
        <p className="label">Quick examples</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map(({ label, text }) => (
            <button
              key={label}
              onClick={() => setContent(text)}
              className="text-xs px-3 py-1.5 rounded-full transition-all duration-150"
              style={{
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border)',
                color: 'var(--text-2)',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(139,92,246,0.4)'; e.currentTarget.style.color = '#c4b5fd' }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-2)' }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="space-y-3">
        <textarea
          value={content}
          onChange={e => setContent(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && (e.metaKey || e.ctrlKey) && moderate()}
          placeholder="Paste or type content to moderate… (Ctrl+Enter to submit)"
          rows={5}
          className="input resize-none"
          style={{ fontFamily: 'inherit', lineHeight: 1.6 }}
        />
        <div className="flex items-center justify-between">
          <span className="text-xs" style={{ color: 'var(--text-3)' }}>{content.length} chars</span>
          <button onClick={moderate} disabled={loading || !content.trim()} className="btn-primary">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
            {loading ? 'Moderating…' : 'Moderate'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div
          className="rounded-xl px-4 py-3 text-sm"
          style={{ background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)', color: '#fca5a5' }}
        >
          {error}
        </div>
      )}

      {/* Result */}
      {result && d && (
        <div
          className="rounded-xl p-5 space-y-5"
          style={{
            background: d.bg,
            border: `1px solid ${d.border}`,
            boxShadow: `0 0 32px ${d.glow}`,
          }}
        >
          {/* Decision */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: `${d.color}18`, border: `1px solid ${d.color}30` }}
              >
                <d.icon size={16} style={{ color: d.color }} />
              </div>
              <span className="font-bold text-base" style={{ color: d.color }}>{d.label}</span>
            </div>
            <span className="text-xs font-mono" style={{ color: 'var(--text-3)' }}>{result.latency_ms}ms</span>
          </div>

          {/* Score bar */}
          <ScoreBar value={result.compliance_score} />

          {/* Reward */}
          <div
            className="flex items-center justify-between text-sm py-2 px-3 rounded-lg"
            style={{ background: 'rgba(0,0,0,0.2)' }}
          >
            <span style={{ color: 'var(--text-3)' }}>RL Reward Signal</span>
            <span className="font-mono font-semibold" style={{ color: 'var(--text-1)' }}>{result.reward.toFixed(3)}</span>
          </div>

          {/* Violations */}
          {result.violations.length > 0 && (
            <div className="space-y-2">
              <p className="label">Violations Detected</p>
              <div className="flex flex-wrap gap-1.5">
                {result.violations.map(v => <ViolationBadge key={v} type={v} />)}
              </div>
            </div>
          )}

          {/* Reasoning */}
          <div className="space-y-2">
            <p className="label">Agent Reasoning</p>
            <p
              className="text-sm leading-relaxed rounded-lg px-3 py-2.5"
              style={{ background: 'rgba(0,0,0,0.25)', color: 'var(--text-2)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              {result.reasoning}
            </p>
          </div>

          {/* Suggested edit */}
          {result.edited_content && (
            <div className="space-y-2">
              <p className="label">Suggested Edit</p>
              <p
                className="text-sm leading-relaxed rounded-lg px-3 py-2.5 italic"
                style={{ background: 'rgba(251,191,36,0.06)', color: '#fde68a', border: '1px solid rgba(251,191,36,0.15)' }}
              >
                {result.edited_content}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
