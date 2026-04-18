import React, { useState } from 'react'
import { Send, Loader2, CheckCircle2, XCircle, Pencil, ChevronRight } from 'lucide-react'
import ViolationBadge from '../components/ViolationBadge'
import { api } from '../hooks/useApi'

const EXAMPLES = [
  { label: 'Clean welcome',    text: 'Hello! Welcome to our community. Looking forward to connecting!' },
  { label: 'Obvious spam',     text: 'BUY NOW!!! Limited offer!!! Free prizes! bit.ly/scam' },
  { label: 'Hate speech',      text: 'I hate those people. They should all die!' },
  { label: 'Subtle harassment',text: 'Some people are just too ignorant to understand these topics. Sad.' },
  { label: 'Health misinfo',   text: "This supplement cured my diabetes in 3 days! Doctors don't want you to know. bit.ly/cure" },
  { label: 'Sarcasm trap',     text: 'I will kill them with my cooking 😂 — my chili is dangerously good.' },
]

const DECISION_CONFIG = {
  approve: { icon: CheckCircle2, color: 'text-emerald-400', label: 'Approved',  border: 'border-emerald-500/20', bg: 'bg-emerald-500/5' },
  reject:  { icon: XCircle,      color: 'text-red-400',     label: 'Rejected',  border: 'border-red-500/20',     bg: 'bg-red-500/5'     },
  edit:    { icon: Pencil,        color: 'text-amber-400',   label: 'Edit',      border: 'border-amber-500/20',   bg: 'bg-amber-500/5'   },
}

function ScoreBar({ value }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.7 ? 'bg-emerald-500' : value >= 0.4 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-500">
        <span>Compliance Score</span>
        <span className="font-mono text-slate-300">{value.toFixed(2)}</span>
      </div>
      <div className="h-1.5 bg-surface-border rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
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
    } finally {
      setLoading(false)
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) moderate()
  }

  const d = result ? DECISION_CONFIG[result.decision] : null

  return (
    <div className="p-8 max-w-3xl space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-slate-100">Playground</h1>
        <p className="text-sm text-slate-500 mt-0.5">Test content through the full 4-step RL moderation pipeline</p>
      </div>

      {/* Examples */}
      <div className="space-y-2">
        <p className="section-title">Quick examples</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map(({ label, text }) => (
            <button
              key={label}
              onClick={() => setContent(text)}
              className="text-xs px-3 py-1.5 rounded-full bg-surface-card hover:bg-surface-hover
                         text-slate-400 hover:text-slate-200 transition-colors border border-surface-border"
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
          onKeyDown={handleKey}
          placeholder="Paste or type content to moderate… (Ctrl+Enter to submit)"
          rows={5}
          className="input resize-none font-sans"
        />
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-600">{content.length} characters</span>
          <button onClick={moderate} disabled={loading || !content.trim()} className="btn-primary">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
            {loading ? 'Moderating…' : 'Moderate'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="card border-red-500/20 bg-red-500/5 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Result */}
      {result && d && (
        <div className={`card border ${d.border} ${d.bg} p-5 space-y-5`}>
          {/* Decision header */}
          <div className="flex items-center justify-between">
            <div className={`flex items-center gap-2 ${d.color}`}>
              <d.icon size={18} />
              <span className="font-bold">{d.label}</span>
            </div>
            <span className="text-xs text-slate-500 font-mono">{result.latency_ms}ms</span>
          </div>

          {/* Score bar */}
          <ScoreBar value={result.compliance_score} />

          {/* Reward */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-500">RL Reward</span>
            <span className="font-mono text-slate-200">{result.reward.toFixed(3)}</span>
          </div>

          {/* Violations */}
          {result.violations.length > 0 && (
            <div className="space-y-2">
              <p className="section-title">Violations Detected</p>
              <div className="flex flex-wrap gap-1.5">
                {result.violations.map(v => <ViolationBadge key={v} type={v} />)}
              </div>
            </div>
          )}

          {/* Reasoning */}
          <div className="space-y-1.5">
            <p className="section-title">Agent Reasoning</p>
            <p className="text-sm text-slate-300 bg-surface-card rounded-lg px-3 py-2.5 border border-surface-border leading-relaxed">
              {result.reasoning}
            </p>
          </div>

          {/* Suggested edit */}
          {result.edited_content && (
            <div className="space-y-1.5">
              <p className="section-title">Suggested Edit</p>
              <p className="text-sm text-slate-300 bg-surface-card rounded-lg px-3 py-2.5 border border-amber-500/20 italic leading-relaxed">
                {result.edited_content}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
