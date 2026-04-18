import React from 'react'
import { CheckCircle, XCircle, Edit3, ThumbsUp, ThumbsDown } from 'lucide-react'
import ViolationBadge from './ViolationBadge'
import { api } from '../hooks/useApi'

const DECISION_STYLE = {
  approve: { icon: CheckCircle, color: 'text-green-400',  bg: 'bg-green-500/10 border-green-500/20' },
  reject:  { icon: XCircle,     color: 'text-red-400',    bg: 'bg-red-500/10 border-red-500/20'   },
  edit:    { icon: Edit3,        color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/20' },
}

export default function ContentCard({ log, onFeedback }) {
  const d = DECISION_STYLE[log.decision] || DECISION_STYLE.approve
  const Icon = d.icon

  async function sendFeedback(correct) {
    await api.post(`/feedback/${log.id}`, null, { params: { correct } })
    onFeedback?.(log.id, correct)
  }

  return (
    <div className={`rounded-xl border p-4 space-y-3 ${d.bg}`}>
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm text-gray-200 leading-relaxed flex-1">{log.content}</p>
        <div className={`flex items-center gap-1.5 text-xs font-semibold ${d.color} shrink-0`}>
          <Icon size={14} />
          {log.decision.toUpperCase()}
        </div>
      </div>

      {log.violations?.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {log.violations.map(v => <ViolationBadge key={v} type={v} />)}
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>Score: <span className="text-gray-300 font-mono">{log.score?.toFixed(2)}</span></span>
        <span>Reward: <span className="text-gray-300 font-mono">{log.reward?.toFixed(3)}</span></span>
        <span>{new Date(log.created_at).toLocaleTimeString()}</span>
        <div className="flex gap-2">
          {log.correct === null || log.correct === undefined ? (
            <>
              <button onClick={() => sendFeedback(true)}  className="hover:text-green-400 transition-colors"><ThumbsUp size={13} /></button>
              <button onClick={() => sendFeedback(false)} className="hover:text-red-400 transition-colors"><ThumbsDown size={13} /></button>
            </>
          ) : (
            <span className={log.correct ? 'text-green-400' : 'text-red-400'}>
              {log.correct ? '✓ Correct' : '✗ Wrong'}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
