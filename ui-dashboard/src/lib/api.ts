import type { Observation, Reward, StepResult } from '@/types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:7860'

export async function checkHealth(): Promise<{
  status: string
  uptime_seconds: number
  episode_count: number
  step_count: number
}> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) throw new Error('Health check failed')
  return response.json()
}

export async function resetEnvironment(
  maxSteps: number = 5,
  difficulty: string = 'mixed'
): Promise<{ observation: Observation; info: Record<string, unknown> }> {
  const response = await fetch(`${API_BASE_URL}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ max_steps: maxSteps, difficulty }),
  })
  if (!response.ok) throw new Error('Reset failed')
  return response.json()
}

export async function stepEnvironment(
  actionType: 'approve' | 'reject' | 'edit',
  editedContent?: string
): Promise<StepResult> {
  const response = await fetch(`${API_BASE_URL}/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action_type: actionType,
      edited_content: editedContent
    }),
  })
  if (!response.ok) throw new Error('Step failed')
  return response.json()
}

export async function evaluateContent(content: string): Promise<{
  score: number
  violations: string[]
  reasoning: string
  source: string
  latency: number
}> {
  const startTime = Date.now()

  // Try OpenAI evaluator first
  try {
    const response = await fetch(`${API_BASE_URL}/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    })

    if (response.ok) {
      const data = await response.json()
      return {
        ...data,
        latency: Date.now() - startTime,
      }
    }
  } catch {
    // Fall through to mock evaluation
  }

  // Mock evaluation for demo
  await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000))

  const hasViolations = Math.random() > 0.5
  const violationTypes = ['hate_speech', 'spam', 'misinformation', 'adult_content', 'violence']

  return {
    score: Math.random(),
    violations: hasViolations
      ? violationTypes.slice(0, Math.floor(Math.random() * 3) + 1)
      : [],
    reasoning: hasViolations
      ? 'The content contains potentially problematic language that may violate community guidelines. Specifically, there are indicators of spam-like behavior and possible misinformation.'
      : 'The content appears to be safe and compliant with community guidelines. No significant violations detected.',
    source: 'mock',
    latency: Date.now() - startTime,
  }
}

export async function getSpec(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE_URL}/spec`)
  if (!response.ok) throw new Error('Failed to get spec')
  return response.json()
}
