export interface Observation {
  content: string
  violations: string[]
  score: number
  step_count: number
}

export interface Reward {
  value: number
  raw_value: number | null
  explanation: string | null
}

export interface StepResult {
  observation: Observation
  reward: Reward
  done: boolean
  truncated: boolean
  info: Record<string, unknown>
}

export interface EvaluationResult {
  score: number
  violations: Violation[]
  reasoning: string
  source: 'openai' | 'fallback' | 'mock'
  latency: number
  confidence: number
  timestamp: string
  content: string
}

export interface Violation {
  type: string
  severity: 'low' | 'medium' | 'high'
  description: string
}

export interface HistoryItem {
  id: string
  content: string
  result: EvaluationResult
  timestamp: string
}

export interface TaskLog {
  step: number
  action: string
  reward: number
  source: string
  timestamp: string
}

export interface AppState {
  isEvaluating: boolean
  currentResult: EvaluationResult | null
  history: HistoryItem[]
  taskLogs: TaskLog[]
  isSidebarOpen: boolean
  isDarkMode: boolean
  apiStatus: 'connected' | 'disconnected' | 'connecting'
  apiLatency: number
}
