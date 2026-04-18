import React, { useState } from 'react'
import { Send, Loader2, CheckCircle, XCircle, Edit3 } from 'lucide-react'
import ViolationBadge from '../components/ViolationBadge'
import { api } from '../hooks/useApi'

const EXAMPLES = [
  "Hello! Welcome to our community. Looking forward to connecting!",
  "BUY NOW!!! Limited offer!!! Free prizes! bit.ly/scam",
  "I hate those people. They should all die!",
  "Some people are just too ignorant to understand these topics. Sad.",
  "This supplement cured my diabetes in 3 days! Doctors don't want you to know. bit.ly/cure",
  "I will kill them with my cooking 😂 — my chili is dangerously good.",
]

const DECISION_CONFIG = {
  approve: { icon: CheckCircle, color: 'text-green-400', label: 'APPROVED',  bg: 'bg-green-500/10 border-green-500/30' },
  reject:  { icon: XCircle,     color: 'text-red-400',   label: 'REJECTED',  bg: 'bg-red-500/10 border-red-500/30'    },
  edit:    { icon: Edit3,        color: 'text-yellow-400',label: 'EDIT',      bg: 'bg-yellow-500/10 border-yellow-500/30' },
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
      setError(e.response?.data?.detail || 'Moderation failed. Check your API key.')
    } finally {
      setLoading(false)
    }
  }

  const d = result ? DECISION_CONFIG[result.decision] : null

  return (
    <div className="p-8 max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Playground</h1>
        <p className="text-gray-400 text-sm mt-1">Test content through the full 4-step RL moderation pipeline</p>
      </div>

      {/* Examples */}
      <div>
        <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide">Quick examples</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map(ex => (
            <button
              key={ex}
              onClick={() => setContent(ex)}
              className="text-xs px-3 py-1.5 rounded-full bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors border border-gray-700"
            >
              {ex.slice(0, 40)}{ex.length > 40 ? '…' : ''}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="space-y-3">
        <textarea
          value={content}
          onChange={e => setContent(e.target.value)}
          placeholder="Paste or type content to moderate..."
          rows={5}
          className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-brand-500 resize-none"
        />
        <button
          onClick={moderate}
          disabled={loading || !content.trim()}
          className="flex items-center gap-2 px-5 py-2.5 bg-brand-500 hover:bg-brand-600 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition-colors"
        >
          {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
          {loading ? 'Moderating…' : 'Moderate'}
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Result */}
      {result && d && (
        <div className={`rounded-xl border p-5 space-y-4 ${d.bg}`}>
          <div className="flex items-center gap-2">
            <d.icon size={20} className={d.color} />
            <span className={`text-lg font-bold ${d.color}`}>{d.label}</span>
            <span className="ml-auto text-xs text-gray-500">{result.latency_ms}ms</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-gray-900/60 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">Compliance Score</p>
              <p className="text-xl font-mono font-bold text-gray-100">{result.compliance_score.toFixed(2)}</p>
            </div>
            <div className="bg-gray-900/60 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">RL Reward</p>
              <p className="text-xl font-mono font-bold text-gray-100">{result.reward.toFixed(3)}</p>
            </div>
          </div>

          {result.violations.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 mb-2">Violations Detected</p>
              <div className="flex flex-wrap gap-1.5">
                {result.violations.map(v => <ViolationBadge key={v} type={v} />)}
              </div>
            </div>
          )}

          <div>
            <p className="text-xs text-gray-500 mb-1">Agent Reasoning</p>
            <p className="text-sm text-gray-300 bg-gray-900/60 rounded-lg p-3">{result.reasoning}</p>
          </div>

          {result.edited_content && (
            <div>
              <p className="text-xs text-gray-500 mb-1">Suggested Edit</p>
              <p className="text-sm text-gray-300 bg-gray-900/60 rounded-lg p-3 italic">{result.edited_content}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
