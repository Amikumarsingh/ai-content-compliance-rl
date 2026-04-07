import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

export function getScoreColor(score: number): string {
  if (score >= 0.7) return 'text-emerald-400'
  if (score >= 0.4) return 'text-amber-400'
  return 'text-red-400'
}

export function getScoreBgColor(score: number): string {
  if (score >= 0.7) return 'bg-emerald-500'
  if (score >= 0.4) return 'bg-amber-500'
  return 'bg-red-500'
}

export function getScoreGlow(score: number): string {
  if (score >= 0.7) return 'safe-glow'
  if (score >= 0.4) return 'warning-glow'
  return 'danger-glow'
}

export function getViolationIconColor(type: string): string {
  const colors: Record<string, string> = {
    hate_speech: 'text-red-400 bg-red-500/10 border-red-500/30',
    harassment: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
    spam: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
    misinformation: 'text-purple-400 bg-purple-500/10 border-purple-500/30',
    adult_content: 'text-pink-400 bg-pink-500/10 border-pink-500/30',
    violence: 'text-red-400 bg-red-500/10 border-red-500/30',
    self_harm: 'text-indigo-400 bg-indigo-500/10 border-indigo-500/30',
  }
  return colors[type] || 'text-gray-400 bg-gray-500/10 border-gray-500/30'
}
