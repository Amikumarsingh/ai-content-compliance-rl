'use client'

import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { useAppStore } from '@/lib/store'
import { evaluateContent } from '@/lib/api'
import { toast } from 'sonner'
import {
  Sparkles,
  Eraser,
  FileText,
  Loader2,
} from 'lucide-react'
import type { EvaluationResult, Violation } from '@/types'

const MAX_CHARS = 5000

interface ContentInputProps {
  onEvaluateComplete?: (result: EvaluationResult) => void
}

export function ContentInput({ onEvaluateComplete }: ContentInputProps) {
  const [content, setContent] = useState('')
  const { setEvaluating, setCurrentResult, addToHistory, isDarkMode } = useAppStore()
  const isEvaluating = useAppStore((state) => state.isEvaluating)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const charCount = content.length
  const charPercentage = (charCount / MAX_CHARS) * 100

  const handleEvaluate = async () => {
    if (!content.trim()) {
      toast.error('Please enter content to evaluate')
      return
    }

    setEvaluating(true)

    try {
      const result = await evaluateContent(content)

      const evaluationResult: EvaluationResult = {
        score: result.score,
        violations: result.violations.map((type) => ({
          type,
          severity: result.score < 0.4 ? 'high' : result.score < 0.7 ? 'medium' : 'low',
          description: getViolationDescription(type),
        })),
        reasoning: result.reasoning,
        source: result.source as 'openai' | 'fallback' | 'mock',
        latency: result.latency,
        confidence: 0.85 + Math.random() * 0.1,
        timestamp: new Date().toISOString(),
        content,
      }

      setCurrentResult(evaluationResult)
      addToHistory({
        id: crypto.randomUUID(),
        content,
        result: evaluationResult,
        timestamp: new Date().toISOString(),
      })

      onEvaluateComplete?.(evaluationResult)
      toast.success('Evaluation Complete', {
        description: `Score: ${evaluationResult.score.toFixed(2)}`,
      })
    } catch (error) {
      toast.error('Evaluation Failed', {
        description: error instanceof Error ? error.message : 'Unknown error occurred',
      })
    } finally {
      setEvaluating(false)
    }
  }

  const handleClear = () => {
    setContent('')
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  const handleLoadExample = (type: 'safe' | 'spam' | 'hate') => {
    const examples = {
      safe: "Thank you for your inquiry! Our team will respond within 24 hours. We appreciate your patience and look forward to assisting you with your questions.",
      spam: "CONGRATULATIONS!!! You've WON $1,000,000! Click HERE now to claim your prize! LIMITED TIME OFFER! Act NOW!!!",
      hate: "People from certain backgrounds are inferior and should be treated differently. They don't deserve the same opportunities.",
    }
    setContent(examples[type])
    toast.info(`Loaded ${type} example`)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Input Card */}
      <div className={cn(
        'relative p-6 rounded-2xl border transition-all duration-300',
        'glass-card',
        isEvaluating && 'border-violet-500/50 glow-effect'
      )}>
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-violet-400" />
            <h2 className="font-semibold text-lg">Content Input</h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Examples:</span>
            <button
              onClick={() => handleLoadExample('safe')}
              className="px-2 py-1 text-xs rounded-md bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 transition-colors"
            >
              Safe
            </button>
            <button
              onClick={() => handleLoadExample('spam')}
              className="px-2 py-1 text-xs rounded-md bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 transition-colors"
            >
              Spam
            </button>
            <button
              onClick={() => handleLoadExample('hate')}
              className="px-2 py-1 text-xs rounded-md bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
            >
              Violation
            </button>
          </div>
        </div>

        {/* Textarea */}
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Paste content to evaluate for compliance..."
            className={cn(
              'w-full h-48 p-4 rounded-xl resize-none',
              'bg-white/5 border border-white/10',
              'focus:outline-none focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/50',
              'transition-all duration-200',
              'text-foreground placeholder:text-muted-foreground/50'
            )}
            disabled={isEvaluating}
            maxLength={MAX_CHARS}
          />

          {/* Character Counter */}
          <div className="absolute bottom-3 right-3 flex items-center gap-2">
            <div className="w-24 h-1.5 rounded-full bg-white/10 overflow-hidden">
              <motion.div
                className={cn(
                  'h-full rounded-full',
                  charPercentage > 90 ? 'bg-red-500' :
                  charPercentage > 70 ? 'bg-amber-500' : 'bg-violet-500'
                )}
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(charPercentage, 100)}%` }}
                transition={{ duration: 0.2 }}
              />
            </div>
            <span className={cn(
              'text-xs font-medium',
              charCount > MAX_CHARS * 0.9 ? 'text-red-400' : 'text-muted-foreground'
            )}>
              {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/10">
          <p className="text-xs text-muted-foreground">
            AI-powered content compliance evaluation
          </p>
          <div className="flex items-center gap-3">
            <button
              onClick={handleClear}
              disabled={isEvaluating || !content}
              className={cn(
                'px-4 py-2 rounded-xl font-medium text-sm',
                'flex items-center gap-2',
                'transition-all duration-200',
                'hover:bg-white/5 disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              <Eraser className="w-4 h-4" />
              Clear
            </button>
            <button
              onClick={handleEvaluate}
              disabled={isEvaluating || !content}
              className={cn(
                'px-6 py-2 rounded-xl font-medium text-sm',
                'flex items-center gap-2',
                'transition-all duration-200',
                'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white',
                'hover:shadow-lg hover:shadow-violet-500/25 hover:scale-105',
                'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100',
                isEvaluating && 'animate-pulse'
              )}
            >
              {isEvaluating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Evaluating...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Evaluate
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function getViolationDescription(type: string): string {
  const descriptions: Record<string, string> = {
    hate_speech: 'Content that promotes hatred or discrimination',
    harassment: 'Behavior that intimidates or threatens others',
    spam: 'Unsolicited or repetitive promotional content',
    misinformation: 'False or misleading information',
    adult_content: 'Sexually explicit or suggestive material',
    violence: 'Content depicting physical harm or aggression',
    self_harm: 'Content promoting self-injury or suicide',
  }
  return descriptions[type] || 'Potential policy violation'
}
