import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { EvaluationResult, HistoryItem, TaskLog } from '@/types'

interface AppState {
  // Evaluation state
  isEvaluating: boolean
  currentResult: EvaluationResult | null
  history: HistoryItem[]

  // Task simulation
  taskLogs: TaskLog[]
  isTaskRunning: boolean

  // UI state
  isSidebarOpen: boolean
  isDarkMode: boolean

  // API state
  apiStatus: 'connected' | 'disconnected' | 'connecting'
  apiLatency: number

  // Actions
  setEvaluating: (isEvaluating: boolean) => void
  setCurrentResult: (result: EvaluationResult | null) => void
  addToHistory: (item: HistoryItem) => void
  clearHistory: () => void

  addTaskLog: (log: TaskLog) => void
  clearTaskLogs: () => void
  setTaskRunning: (isRunning: boolean) => void

  toggleSidebar: () => void
  toggleDarkMode: () => void

  setApiStatus: (status: 'connected' | 'disconnected' | 'connecting') => void
  setApiLatency: (latency: number) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Initial state
      isEvaluating: false,
      currentResult: null,
      history: [],
      taskLogs: [],
      isTaskRunning: false,
      isSidebarOpen: true,
      isDarkMode: true,
      apiStatus: 'disconnected',
      apiLatency: 0,

      // Actions
      setEvaluating: (isEvaluating) => set({ isEvaluating }),
      setCurrentResult: (result) => set({ currentResult: result }),
      addToHistory: (item) =>
        set((state) => ({
          history: [item, ...state.history].slice(0, 50)
        })),
      clearHistory: () => set({ history: [] }),

      addTaskLog: (log) =>
        set((state) => ({
          taskLogs: [...state.taskLogs, log].slice(-100)
        })),
      clearTaskLogs: () => set({ taskLogs: [] }),
      setTaskRunning: (isRunning) => set({ isTaskRunning: isRunning }),

      toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
      toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),

      setApiStatus: (status) => set({ apiStatus: status }),
      setApiLatency: (latency) => set({ apiLatency: latency }),
    }),
    {
      name: 'ai-compliance-storage',
      partialize: (state) => ({
        history: state.history,
        isDarkMode: state.isDarkMode
      }),
    }
  )
)
