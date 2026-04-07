'use client'

import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { useAppStore } from '@/lib/store'
import { SidebarToggle } from './Sidebar'
import {
  Wifi,
  WifiOff,
  Gauge,
  Moon,
  Sun,
  Cpu,
} from 'lucide-react'

export function TopBar() {
  const { apiStatus, apiLatency, isDarkMode, toggleDarkMode } = useAppStore()

  return (
    <header className="h-16 glass border-b border-white/10 px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <SidebarToggle />

        {/* Breadcrumb */}
        <nav className="hidden md:flex items-center gap-2 text-sm text-muted-foreground">
          <span className="hover:text-foreground transition-colors cursor-pointer">AI Compliance</span>
          <span>/</span>
          <span className="text-foreground font-medium">Dashboard</span>
        </nav>
      </div>

      <div className="flex items-center gap-4">
        {/* Model Indicator */}
        <motion.div
          className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Cpu className="w-4 h-4 text-violet-400" />
          <span className="text-xs font-medium">gpt-4o-mini</span>
        </motion.div>

        {/* Status Indicator */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-full border',
            apiStatus === 'connected' && 'bg-emerald-500/10 border-emerald-500/30',
            apiStatus === 'connecting' && 'bg-amber-500/10 border-amber-500/30',
            apiStatus === 'disconnected' && 'bg-red-500/10 border-red-500/30'
          )}
        >
          {apiStatus === 'connected' ? (
            <Wifi className="w-4 h-4 text-emerald-400" />
          ) : apiStatus === 'connecting' ? (
            <Wifi className="w-4 h-4 text-amber-400 animate-pulse" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-400" />
          )}
          <span className={cn(
            'text-xs font-medium capitalize',
            apiStatus === 'connected' && 'text-emerald-400',
            apiStatus === 'connecting' && 'text-amber-400',
            apiStatus === 'disconnected' && 'text-red-400'
          )}>
            {apiStatus}
          </span>
        </motion.div>

        {/* Latency Badge */}
        <motion.div
          className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Gauge className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-medium">{apiLatency}ms</span>
        </motion.div>

        {/* Theme Toggle */}
        <button
          onClick={toggleDarkMode}
          className="p-2 rounded-lg hover:bg-white/5 transition-colors"
        >
          {isDarkMode ? (
            <Sun className="w-5 h-5" />
          ) : (
            <Moon className="w-5 h-5" />
          )}
        </button>
      </div>
    </header>
  )
}
