import React from 'react'
import { CheckCircle2, XCircle, Pencil, ThumbsUp, ThumbsDown } from 'lucide-react'
import ViolationBadge from './ViolationBadge'
import { api } from '../hooks/useApi'

const DECISION = {
  approve: {
    icon: CheckCircle2,
    label: 'Approved',
    color: '#34d399',
    bg: 'rgba(52,211,153,0.06)',
    border: 'rgba(52,211,153,0.15)',
  },
  reject: {
    icon: XCircle,
    label: 'Rejected',
    color: '#f87171',
    bg: 'rgba(248,113,113,0.06)',
    border: 'rgba(248,113,113,0.15)',
  },
  edit: {
    icon: Pencil,
    label: 'Edit',
    color: '#fbbf24',
    bg: 'rgba(251,191,36,0.06)',
    border: 'rgba(251,191,36,0.15)',
  },
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
    <div
      className="rounded-xl p-4 space-y-3 transition-all duration-150"
      style={{ background: d.bg, border: `1px solid ${d.border}` }}
    >
      {/* Top row */}
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm leading-relaxed flex-1" style={{ color: 'var(--text-1)' }}>
          {log.content}
        </p>
        <div className="flex items-center gap-1.5 shrink-0" style={{ color: d.color }}>
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
      <div
        className="flex items-center gap-4 text-xs pt-2"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)', color: 'var(--text-3)' }}
      >
        <span>Score <span className="font-mono" style={{ color: 'var(--text-2)' }}>{log.score?.toFixed(2)}</span></span>
        <span>Reward <span className="font-mono" style={{ color: 'var(--text-2)' }}>{log.reward?.toFixed(3)}</span></span>
        <span className="ml-auto">{new Date(log.created_at).toLocaleTimeString()}</span>
        <div className="flex items-center gap-2">
          {log.correct == null ? (
            <>
              <button onClick={() => sendFeedback(true)}  className="hover:text-emerald-400 transition-colors" title="Correct"><ThumbsUp size={12} /></button>
              <button onClick={() => sendFeedback(false)} className="hover:text-red-400 transition-colors"     title="Wrong"><ThumbsDown size={12} /></button>
            </>
          ) : (
            <span style={{ color: log.correct ? '#34d399' : '#f87171' }}>
              {log.correct ? '✓ Correct' : '✗ Wrong'}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
