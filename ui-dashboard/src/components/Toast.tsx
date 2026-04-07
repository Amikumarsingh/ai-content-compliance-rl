'use client'

import { Toaster } from 'sonner'
import { useAppStore } from '@/lib/store'

export function Toast() {
  const isDarkMode = useAppStore((state) => state.isDarkMode)

  return (
    <Toaster
      position="bottom-right"
      theme={isDarkMode ? 'dark' : 'light'}
      toastOptions={{
        classNames: {
          toast: 'glass-card border border-white/10',
          title: 'text-foreground',
          description: 'text-muted-foreground',
          success: 'border-emerald-500/30',
          error: 'border-red-500/30',
          warning: 'border-amber-500/30',
          info: 'border-violet-500/30',
        },
      }}
      expand
      visibleToasts={5}
    />
  )
}
