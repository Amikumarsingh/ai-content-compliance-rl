'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { useAppStore } from '@/lib/store'
import {
  LayoutDashboard,
  ShieldCheck,
  ClipboardList,
  History,
  Settings,
  ChevronLeft,
  ChevronRight,
  Zap,
  BookOpen,
  AlertTriangle,
} from 'lucide-react'

const menuItems = [
  { icon: LayoutDashboard, label: 'Dashboard', id: 'dashboard' },
  { icon: ShieldCheck, label: 'Evaluate', id: 'evaluate' },
  {
    icon: ClipboardList,
    label: 'Tasks',
    id: 'tasks',
    children: [
      { icon: Zap, label: 'Easy', id: 'easy' },
      { icon: BookOpen, label: 'Medium', id: 'medium' },
      { icon: AlertTriangle, label: 'Hard', id: 'hard' },
    ],
  },
  { icon: History, label: 'History', id: 'history' },
  { icon: Settings, label: 'Settings', id: 'settings' },
]

export function Sidebar() {
  const { isSidebarOpen, toggleSidebar, isDarkMode } = useAppStore()

  return (
    <AnimatePresence mode="wait">
      {isSidebarOpen ? (
        <motion.aside
          key="sidebar-open"
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 280, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          className={cn(
            'relative h-full overflow-hidden',
            'glass border-r border-white/10'
          )}
        >
          <div className="p-6 h-full flex flex-col">
            {/* Logo */}
            <div className="flex items-center gap-3 mb-8">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center glow-effect">
                <ShieldCheck className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-lg text-gradient">AI Compliance</h1>
                <p className="text-xs text-muted-foreground">RL Environment</p>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 space-y-2">
              {menuItems.map((item) => (
                <div key={item.id}>
                  <button
                    className={cn(
                      'w-full flex items-center gap-3 px-4 py-3 rounded-xl',
                      'transition-all duration-200 group',
                      'hover:bg-white/5 hover:translate-x-1',
                      'text-muted-foreground hover:text-foreground'
                    )}
                  >
                    <item.icon className="w-5 h-5 transition-transform group-hover:scale-110" />
                    <span className="font-medium">{item.label}</span>
                  </button>

                  {item.children && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      className="ml-4 pl-4 border-l border-white/10 space-y-1"
                    >
                      {item.children.map((child) => (
                        <button
                          key={child.id}
                          className={cn(
                            'w-full flex items-center gap-3 px-4 py-2 rounded-lg',
                            'transition-all duration-200 group',
                            'hover:bg-white/5 hover:translate-x-1',
                            'text-sm text-muted-foreground hover:text-foreground'
                          )}
                        >
                          <child.icon className="w-4 h-4" />
                          <span>{child.label}</span>
                        </button>
                      ))}
                    </motion.div>
                  )}
                </div>
              ))}
            </nav>

            {/* Footer */}
            <div className="pt-4 border-t border-white/10">
              <div className="p-4 rounded-xl bg-gradient-to-br from-violet-500/10 to-fuchsia-500/10 border border-violet-500/20">
                <p className="text-xs text-muted-foreground mb-2">Version</p>
                <p className="text-sm font-semibold text-foreground">1.0.0</p>
              </div>
            </div>
          </div>
        </motion.aside>
      ) : (
        <motion.aside
          key="sidebar-closed"
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 80, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          className={cn(
            'relative h-full overflow-hidden',
            'glass border-r border-white/10'
          )}
        >
          <div className="p-4 h-full flex flex-col items-center gap-4">
            {/* Logo Mini */}
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center glow-effect">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>

            {/* Mini Nav */}
            {menuItems.map((item) => (
              <button
                key={item.id}
                className="w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 hover:bg-white/5 group"
                title={item.label}
              >
                <item.icon className="w-5 h-5 text-muted-foreground group-hover:text-foreground" />
              </button>
            ))}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}

export function SidebarToggle() {
  const { isSidebarOpen, toggleSidebar } = useAppStore()

  return (
    <button
      onClick={toggleSidebar}
      className="p-2 rounded-lg hover:bg-white/5 transition-colors"
    >
      {isSidebarOpen ? (
        <ChevronLeft className="w-5 h-5" />
      ) : (
        <ChevronRight className="w-5 h-5" />
      )}
    </button>
  )
}
