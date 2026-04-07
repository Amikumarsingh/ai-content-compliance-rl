'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { cn, getScoreColor, getScoreBgColor } from '@/lib/utils'
import { useAppStore } from '@/lib/store'
import {
  History,
  Trash2,
  ExternalLink,
  Clock,
  Search,
  X,
  Download,
} from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

export function HistoryPanel() {
  const { history, clearHistory } = useAppStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedItem, setSelectedItem] = useState<typeof history[0] | null>(null)

  const filteredHistory = history.filter((item) =>
    item.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.result.violations.some(v => v.type.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `compliance-history-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('History exported')
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
          <History className="w-5 h-5 text-violet-400" />
          <h2 className="font-semibold text-lg">Evaluation History</h2>
          <span className="text-xs text-muted-foreground px-2 py-0.5 rounded-full bg-white/10">
            {history.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {history.length > 0 && (
            <>
              <button
                onClick={handleExport}
                className="p-2 rounded-lg hover:bg-white/5 transition-colors text-muted-foreground hover:text-foreground"
                title="Export history"
              >
                <Download className="w-4 h-4" />
              </button>
              <button
                onClick={clearHistory}
                className="p-2 rounded-lg hover:bg-white/5 transition-colors text-muted-foreground hover:text-foreground"
                title="Clear history"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search history..."
          className="w-full pl-10 pr-10 py-2.5 rounded-xl bg-white/5 border border-white/10 focus:outline-none focus:border-violet-500/50 text-sm"
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* History List */}
      <div className="space-y-2 max-h-[400px] overflow-y-auto">
        <AnimatePresence mode="popLayout">
          {filteredHistory.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center py-12"
            >
              <History className="w-12 h-12 mx-auto mb-3 text-muted-foreground/50" />
              <p className="text-muted-foreground">
                {history.length === 0 ? 'No evaluations yet' : 'No matching results'}
              </p>
            </motion.div>
          ) : (
            filteredHistory.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -100 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => setSelectedItem(item)}
                className={cn(
                  'p-4 rounded-xl border cursor-pointer transition-all duration-200',
                  'glass-card hover:border-violet-500/50 hover:shadow-lg hover:shadow-violet-500/10',
                  selectedItem?.id === item.id && 'border-violet-500/50'
                )}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-foreground truncate">
                      {item.content}
                    </p>
                    <div className="flex items-center gap-3 mt-2">
                      <span className={cn(
                        'text-xs font-bold',
                        getScoreColor(item.result.score)
                      )}>
                        Score: {item.result.score.toFixed(2)}
                      </span>
                      <span className="text-xs text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </span>
                      {item.result.violations.length > 0 && (
                        <span className="text-xs text-amber-400">
                          {item.result.violations.length} violation{item.result.violations.length > 1 ? 's' : ''}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={cn(
                      'w-3 h-3 rounded-full',
                      getScoreBgColor(item.result.score)
                    )} />
                    <ExternalLink className="w-4 h-4 text-muted-foreground" />
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedItem && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedItem(null)}
          >
            <motion.div
              initial={{ scale: 0.95, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 20 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-2xl max-h-[80vh] overflow-y-auto rounded-2xl glass-card border border-white/10"
            >
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-lg">Evaluation Details</h3>
                  <button
                    onClick={() => setSelectedItem(null)}
                    className="p-2 rounded-lg hover:bg-white/5 transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                  <p className="text-xs text-muted-foreground mb-1">Content</p>
                  <p className="text-sm">{selectedItem.content}</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                    <p className="text-xs text-muted-foreground mb-1">Score</p>
                    <p className={cn('text-2xl font-bold', getScoreColor(selectedItem.result.score))}>
                      {selectedItem.result.score.toFixed(2)}
                    </p>
                  </div>
                  <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                    <p className="text-xs text-muted-foreground mb-1">Source</p>
                    <p className="text-lg font-medium">{selectedItem.result.source}</p>
                  </div>
                </div>

                {selectedItem.result.violations.length > 0 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-2">Violations</p>
                    <div className="flex flex-wrap gap-2">
                      {selectedItem.result.violations.map((v) => (
                        <span
                          key={v.type}
                          className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs font-medium"
                        >
                          {v.type}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <p className="text-xs text-muted-foreground mb-2">Reasoning</p>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {selectedItem.result.reasoning}
                  </p>
                </div>

                <div className="flex items-center justify-between pt-4 border-t border-white/10">
                  <span className="text-xs text-muted-foreground">
                    {new Date(selectedItem.timestamp).toLocaleString()}
                  </span>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(JSON.stringify(selectedItem.result, null, 2))
                      toast.success('Copied to clipboard')
                    }}
                    className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-sm"
                  >
                    Copy JSON
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
