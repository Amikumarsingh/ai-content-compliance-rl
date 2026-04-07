'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { cn, getScoreColor, getScoreBgColor, getScoreGlow, getViolationIconColor } from '@/lib/utils'
import { useAppStore } from '@/lib/store'
import {
  Gauge,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  MessageSquare,
  Clock,
  Database,
  Percent,
  Copy,
  Download,
  Check,
} from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

const violationIcons: Record<string, React.ReactNode> = {
  hate_speech: <ShieldAlert className="w-4 h-4" />,
  harassment: <AlertTriangle className="w-4 h-4" />,
  spam: <AlertTriangle className="w-4 h-4" />,
  misinformation: <MessageSquare className="w-4 h-4" />,
  adult_content: <ShieldAlert className="w-4 h-4" />,
  violence: <ShieldAlert className="w-4 h-4" />,
  self_harm: <ShieldCheck className="w-4 h-4" />,
}

export function ResultCards() {
  const currentResult = useAppStore((state) => state.currentResult)
  const [copiedReasoning, setCopiedReasoning] = useState(false)

  if (!currentResult) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="h-32 rounded-2xl bg-white/5 border border-white/10 animate-pulse"
          />
        ))}
      </motion.div>
    )
  }

  const { score, violations, reasoning, source, latency, confidence } = currentResult

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
    >
      {/* Score Card */}
      <ScoreCard score={score} />

      {/* Violations Card */}
      <ViolationsCard violations={violations} />

      {/* Metadata Card */}
      <MetadataCard source={source} latency={latency} confidence={confidence} />

      {/* Actions Card */}
      <ActionsCard
        reasoning={reasoning}
        onCopy={() => {
          navigator.clipboard.writeText(reasoning)
          setCopiedReasoning(true)
          toast.success('Reasoning copied to clipboard')
          setTimeout(() => setCopiedReasoning(false), 2000)
        }}
        onExport={() => {
          const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' })
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `compliance-result-${Date.now()}.json`
          a.click()
          URL.revokeObjectURL(url)
          toast.success('Result exported as JSON')
        }}
      />

      {/* Reasoning Card - Full Width */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="md:col-span-2 lg:col-span-4"
      >
        <div className={cn(
          'p-6 rounded-2xl border transition-all duration-300',
          'glass-card'
        )}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-violet-400" />
              <h3 className="font-semibold">AI Reasoning</h3>
            </div>
            <button
              onClick={() => {
                navigator.clipboard.writeText(reasoning)
                toast.success('Reasoning copied')
              }}
              className="p-2 rounded-lg hover:bg-white/5 transition-colors text-muted-foreground hover:text-foreground"
            >
              <Copy className="w-4 h-4" />
            </button>
          </div>
          <p className="text-muted-foreground leading-relaxed">
            {reasoning}
          </p>
        </div>
      </motion.div>
    </motion.div>
  )
}

function ScoreCard({ score }: { score: number }) {
  const percentage = Math.round(score * 100)

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        'p-6 rounded-2xl border transition-all duration-300',
        'glass-card',
        getScoreGlow(score)
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-muted-foreground">Compliance Score</span>
        <Gauge className={cn('w-5 h-5', getScoreColor(score))} />
      </div>

      <div className="flex items-end justify-between">
        <div>
          <motion.span
            className={cn('text-4xl font-bold', getScoreColor(score))}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            {score.toFixed(2)}
          </motion.span>
          <p className="text-xs text-muted-foreground mt-1">
            {score >= 0.7 ? 'Safe' : score >= 0.4 ? 'Review Needed' : 'Unsafe'}
          </p>
        </div>
        <div className="text-right">
          <span className="text-2xl font-bold">{percentage}%</span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mt-4 h-2 rounded-full bg-white/10 overflow-hidden">
        <motion.div
          className={cn('h-full rounded-full', getScoreBgColor(score))}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </div>

      {/* Status Indicator */}
      <div className="mt-4 flex items-center gap-2">
        <motion.div
          className={cn(
            'w-2 h-2 rounded-full',
            getScoreBgColor(score),
            'animate-pulse-glow'
          )}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
        />
        <span className="text-xs text-muted-foreground">
          {score >= 0.7 ? 'Content appears safe' : score >= 0.4 ? 'Manual review recommended' : 'Content flagged'}
        </span>
      </div>
    </motion.div>
  )
}

function ViolationsCard({ violations }: { violations: { type: string; severity: string; description: string }[] }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.1 }}
      className={cn(
        'p-6 rounded-2xl border transition-all duration-300',
        'glass-card'
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-muted-foreground">Violations</span>
        <AlertTriangle className={cn(
          'w-5 h-5',
          violations.length > 0 ? 'text-amber-400' : 'text-emerald-400'
        )} />
      </div>

      <div className="flex items-end justify-between">
        <div>
          <motion.span
            className={cn(
              'text-4xl font-bold',
              violations.length > 0 ? 'text-amber-400' : 'text-emerald-400'
            )}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            {violations.length}
          </motion.span>
          <p className="text-xs text-muted-foreground mt-1">
            {violations.length === 0 ? 'No violations' : violations.length === 1 ? 'Violation detected' : 'Violations detected'}
          </p>
        </div>
      </div>

      {/* Violation Badges */}
      <div className="mt-4 flex flex-wrap gap-2">
        <AnimatePresence mode="popLayout">
          {violations.length > 0 ? (
            violations.map((violation, index) => (
              <motion.div
                key={violation.type}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ delay: index * 0.1 }}
                className={cn(
                  'px-3 py-1.5 rounded-lg border flex items-center gap-2',
                  'text-xs font-medium',
                  getViolationIconColor(violation.type)
                )}
              >
                {violationIcons[violation.type] || <AlertTriangle className="w-3 h-3" />}
                {violation.type.replace('_', ' ')}
              </motion.div>
            ))
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium"
            >
              <ShieldCheck className="w-3 h-3 inline mr-1" />
              Clean
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

function MetadataCard({
  source,
  latency,
  confidence
}: {
  source: string
  latency: number
  confidence: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.15 }}
      className={cn(
        'p-6 rounded-2xl border transition-all duration-300',
        'glass-card'
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-muted-foreground">Metadata</span>
        <Database className="w-5 h-5 text-cyan-400" />
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Source</span>
          <span className="text-xs font-medium px-2 py-1 rounded bg-white/10">
            {source}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="w-3 h-3" />
            Latency
          </div>
          <span className="text-xs font-medium">{latency}ms</span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Percent className="w-3 h-3" />
            Confidence
          </div>
          <span className="text-xs font-medium">{(confidence * 100).toFixed(0)}%</span>
        </div>
      </div>
    </motion.div>
  )
}

function ActionsCard({
  reasoning,
  onCopy,
  onExport
}: {
  reasoning: string
  onCopy: () => void
  onExport: () => void
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.2 }}
      className={cn(
        'p-6 rounded-2xl border transition-all duration-300',
        'glass-card'
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-muted-foreground">Actions</span>
        <Check className="w-5 h-5 text-violet-400" />
      </div>

      <div className="space-y-2">
        <button
          onClick={onCopy}
          className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-200 flex items-center justify-between group"
        >
          <div className="flex items-center gap-3">
            <Copy className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
            <span className="text-sm">Copy Reasoning</span>
          </div>
        </button>

        <button
          onClick={onExport}
          className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-200 flex items-center justify-between group"
        >
          <div className="flex items-center gap-3">
            <Download className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
            <span className="text-sm">Export JSON</span>
          </div>
        </button>
      </div>
    </motion.div>
  )
}
