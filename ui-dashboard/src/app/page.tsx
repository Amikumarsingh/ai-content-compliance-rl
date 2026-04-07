'use client'

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useAppStore } from '@/lib/store'
import { checkHealth } from '@/lib/api'
import { Sidebar } from '@/components/Sidebar'
import { TopBar } from '@/components/TopBar'
import { ContentInput } from '@/components/ContentInput'
import { ResultCards } from '@/components/ResultCards'
import { TaskPanel } from '@/components/TaskPanel'
import { HistoryPanel } from '@/components/HistoryPanel'
import { Toast } from '@/components/Toast'
import { toast } from 'sonner'

export default function Dashboard() {
  const { setApiStatus, setApiLatency, isDarkMode } = useAppStore()

  useEffect(() => {
    // Check API health on mount
    const checkApiHealth = async () => {
      setApiStatus('connecting')
      try {
        const startTime = Date.now()
        const health = await checkHealth()
        const latency = Date.now() - startTime

        setApiStatus('connected')
        setApiLatency(latency)
        console.log('API Health:', health)
      } catch (error) {
        setApiStatus('disconnected')
        console.warn('API not available, running in demo mode')
      }
    }

    checkApiHealth()

    // Poll every 30 seconds
    const interval = setInterval(checkApiHealth, 30000)
    return () => clearInterval(interval)
  }, [setApiStatus, setApiLatency])

  return (
    <div className={`h-screen flex ${isDarkMode ? 'dark' : ''}`}>
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <TopBar />

        {/* Scrollable Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            {/* Welcome Section */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8"
            >
              <h1 className="text-3xl font-bold text-gradient mb-2">
                Content Compliance Dashboard
              </h1>
              <p className="text-muted-foreground">
                AI-powered content moderation with reinforcement learning
              </p>
            </motion.div>

            {/* Content Input */}
            <ContentInput />

            {/* Result Cards */}
            <ResultCards />

            {/* Task Simulation */}
            <TaskPanel />

            {/* History */}
            <HistoryPanel />
          </div>
        </main>
      </div>

      {/* Toast Notifications */}
      <Toast />
    </div>
  )
}
