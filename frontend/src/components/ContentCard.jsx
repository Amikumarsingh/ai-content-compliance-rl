import React from 'react'
import { CheckCircle2, XCircle, Pencil, ThumbsUp, ThumbsDown } from 'lucide-react'
import ViolationBadge from './ViolationBadge'
import { api } from '../hooks/useApi'

const DECISION = {
  approve: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-500/8 border-emerald-500/15', label: 'Approved' },
  reject:  { icon: XCircle,      color: 'text-red-400',     bg: 'bg-red-500/8 border-red-500/15',         label: 'Rejected' },
  edit:    { icon: Pencil,        color: 'text-amber-400',   bg: 'bg-amber-500/8 border-amber-500/15',     label: 'Edit'     },
}

export default function ContentCard({ log, onFeedback }) {
  const d = DECISION[log.decision] ?? DECISION.approve
  const Icon = d.icon

  async function sendFeedback(correct) {
    try {
      await api.post(`/feedback/${log.id}`, null, { params: { correct } })
      onFeedback?.(log.id, correct)
    } catch {}
  }

  return (
    <div className={`card border p-4 space-y-3 ${d.bg}`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm text-slate-200 leading-relaxed flex-1 line-clamp-2">{log.content}</p>
        <div className={`flex items-center gap-1.5 shrink-0 ${d.color}`}>
          <Icon size={13} />
          <span className="text-xs font-semibold">{d.label}</span>
        </div>
      </div>

      {/* Violations */}
      {log.violations?.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {log.violations.map(v => <ViolationBadge key={v} type={v} />)}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center gap-4 text-xs text-slate-500 pt-1 border-t border-surface-border">
        <span>Score <span className="font-mono text-slate-300">{log.score?.toFixed(2)}</span></span>
        <span>Reward <span className="font-mono text-slate-300">{log.reward?.toFixed(3)}</span></span>
        <span className="ml-auto">{new Date(log.created_at).toLocaleTimeString()}</span>
        <div className="flex items-center gap-1.5">
          {log.correct == null ? (
            <>
              <button onClick={() => sendFeedback(true)}  title="Correct" className="hover:text-emerald-400 transition-colors"><ThumbsUp size={12} /></button>
              <button onClick={() => sendFeedback(false)} title="Wrong"   className="hover:text-red-400 transition-colors"><ThumbsDown size={12} /></button>
            </>
          ) : (
            <span className={log.correct ? 'text-emerald-400' : 'text-red-400'}>
              {log.correct ? '✓ Correct' : '✗ Wrong'}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
