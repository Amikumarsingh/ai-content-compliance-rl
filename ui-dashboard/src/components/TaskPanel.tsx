'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { useAppStore } from '@/lib/store'
import {
  Zap,
  BookOpen,
  AlertTriangle,
  Play,
  Square,
  Trash2,
  Terminal,
  CheckCircle,
  XCircle,
} from 'lucide-react'
import type { TaskLog } from '@/types'

const taskConfigs = {
  easy: {
    name: 'Easy',
    icon: Zap,
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10',
    borderColor: 'border-emerald-500/30',
    steps: 3,
    actions: ['approve', 'approve', 'reject'],
  },
  medium: {
    name: 'Medium',
    icon: BookOpen,
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/30',
    steps: 5,
    actions: ['approve', 'edit', 'approve', 'reject', 'approve'],
  },
  hard: {
    name: 'Hard',
    icon: AlertTriangle,
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
    steps: 8,
    actions: ['edit', 'approve', 'reject', 'edit', 'approve', 'reject', 'approve', 'edit'],
  },
}

export function TaskPanel() {
  const { taskLogs, addTaskLog, clearTaskLogs, isTaskRunning, setTaskRunning } = useAppStore()
  const [selectedTask, setSelectedTask] = useState<'easy' | 'medium' | 'hard' | null>(null)
  const [currentStep, setCurrentStep] = useState(0)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [taskLogs])

  const runTask = async (difficulty: 'easy' | 'medium' | 'hard') => {
    if (isTaskRunning) return

    const config = taskConfigs[difficulty]
    setSelectedTask(difficulty)
    setTaskRunning(true)
    clearTaskLogs()
    setCurrentStep(0)

    // Add initial log
    addTaskLog({
      step: 0,
      action: 'init',
      reward: 0,
      source: `Starting ${config.name} task...`,
      timestamp: new Date().toISOString(),
    })

    for (let i = 0; i < config.steps; i++) {
      setCurrentStep(i + 1)

      await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 400))

      const action = config.actions[i]
      const reward = Math.random() > 0.3 ? Math.random() * 0.5 + 0.5 : Math.random() * 0.3
      const success = reward > 0.5

      addTaskLog({
        step: i + 1,
        action,
        reward,
        source: `action=${action}, reward=${reward.toFixed(3)}, success=${success}`,
        timestamp: new Date().toISOString(),
      })
    }

    // Completion log
    await new Promise(resolve => setTimeout(resolve, 500))
    addTaskLog({
      step: config.steps + 1,
      action: 'complete',
      reward: 0,
      source: `Task completed! Final score: ${(taskLogs.reduce((sum, log) => sum + log.reward, 0) / config.steps).toFixed(3)}`,
      timestamp: new Date().toISOString(),
    })

    setTaskRunning(false)
    setSelectedTask(null)
  }

  const stopTask = () => {
    setTaskRunning(false)
    addTaskLog({
      step: currentStep,
      action: 'stop',
      reward: 0,
      source: 'Task stopped by user',
      timestamp: new Date().toISOString(),
    })
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-violet-400" />
          <h2 className="font-semibold text-lg">Task Simulation</h2>
        </div>
        <button
          onClick={clearTaskLogs}
          className="p-2 rounded-lg hover:bg-white/5 transition-colors text-muted-foreground hover:text-foreground"
          title="Clear logs"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Task Buttons */}
      <div className="flex gap-3">
        {(Object.entries(taskConfigs) as [keyof typeof taskConfigs, typeof taskConfigs[keyof typeof taskConfigs]][]).map(([key, config]) => (
          <button
            key={key}
            onClick={() => runTask(key)}
            disabled={isTaskRunning}
            className={cn(
              'flex-1 px-4 py-3 rounded-xl border transition-all duration-200',
              'flex items-center justify-center gap-2',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              config.bgColor,
              config.borderColor,
              isTaskRunning && selectedTask === key
                ? 'animate-pulse'
                : 'hover:scale-105 hover:shadow-lg'
            )}
          >
            <config.icon className={cn('w-5 h-5', config.color)} />
            <span className={cn('font-medium', config.color)}>
              {config.name}
            </span>
          </button>
        ))}

        {isTaskRunning && (
          <button
            onClick={stopTask}
            className="px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 transition-all duration-200"
          >
            <Square className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Terminal Output */}
      <div className={cn(
        'relative rounded-2xl border overflow-hidden',
        'bg-black/50 border-white/10',
        'min-h-[300px] max-h-[400px]'
      )}>
        {/* Terminal Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-white/5">
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500/80" />
              <div className="w-3 h-3 rounded-full bg-amber-500/80" />
              <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
            </div>
            <span className="text-xs text-muted-foreground ml-2">RL Environment Logs</span>
          </div>
          {isTaskRunning && (
            <div className="flex items-center gap-2 text-xs text-emerald-400">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Running
            </div>
          )}
        </div>

        {/* Terminal Content */}
        <div className="p-4 font-mono text-xs space-y-2 overflow-y-auto max-h-[280px]">
          <AnimatePresence>
            {taskLogs.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-muted-foreground text-center py-8"
              >
                <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>Click a task button to start simulation</p>
              </motion.div>
            ) : (
              taskLogs.map((log, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-start gap-3"
                >
                  <span className="text-muted-foreground shrink-0">
                    [{new Date(log.timestamp).toLocaleTimeString()}]
                  </span>
                  <span className={cn(
                    'shrink-0 font-bold',
                    log.action === 'approve' && 'text-emerald-400',
                    log.action === 'reject' && 'text-red-400',
                    log.action === 'edit' && 'text-amber-400',
                    log.action === 'init' && 'text-violet-400',
                    log.action === 'complete' && 'text-cyan-400',
                    log.action === 'stop' && 'text-orange-400'
                  )}>
                    {log.action.toUpperCase()}
                  </span>
                  <span className="text-foreground">
                    {log.source}
                  </span>
                  {log.reward > 0 && (
                    <span className={cn(
                      'shrink-0 font-mono',
                      log.reward > 0.5 ? 'text-emerald-400' : 'text-amber-400'
                    )}>
                      R: {log.reward.toFixed(3)}
                    </span>
                  )}
                  {log.action === 'approve' && <CheckCircle className="w-3 h-3 text-emerald-400 shrink-0" />}
                  {log.action === 'reject' && <XCircle className="w-3 h-3 text-red-400 shrink-0" />}
                </motion.div>
              ))
            )}
          </AnimatePresence>
          <div ref={logsEndRef} />
        </div>

        {/* Cursor */}
        {isTaskRunning && (
          <div className="px-4 py-2 border-t border-white/10 bg-white/5">
            <span className="text-emerald-400 font-mono text-xs">
              <span className="text-muted-foreground">$</span> Processing<span className="terminal-cursor">_</span>
            </span>
          </div>
        )}
      </div>

      {/* Progress Indicator */}
      {isTaskRunning && selectedTask && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={cn(
            'px-4 py-3 rounded-xl border flex items-center justify-between',
            taskConfigs[selectedTask].bgColor,
            taskConfigs[selectedTask].borderColor
          )}
        >
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            >
              <Zap className={cn('w-5 h-5', taskConfigs[selectedTask].color)} />
            </motion.div>
            <div>
              <p className={cn('font-medium', taskConfigs[selectedTask].color)}>
                Running {taskConfigs[selectedTask].name} Task
              </p>
              <p className="text-xs text-muted-foreground">
                Step {currentStep} / {taskConfigs[selectedTask].steps}
              </p>
            </div>
          </div>
          <div className="w-32 h-2 rounded-full bg-white/10 overflow-hidden">
            <motion.div
              className={cn('h-full rounded-full', taskConfigs[selectedTask].color.replace('text-', 'bg-'))}
              initial={{ width: 0 }}
              animate={{ width: `${(currentStep / taskConfigs[selectedTask].steps) * 100}%` }}
            />
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
