import React, { useEffect, useState } from 'react'
import { Plus, Trash2, Copy, Check, Key, AlertTriangle, Zap } from 'lucide-react'
import { api, setApiKey } from '../hooks/useApi'

const PLAN_STYLE = {
  free:       { bg: 'rgba(100,116,139,0.12)', color: '#94a3b8', border: 'rgba(100,116,139,0.25)', label: 'Free · 1k'    },
  pro:        { bg: 'rgba(139,92,246,0.12)',  color: '#c4b5fd', border: 'rgba(139,92,246,0.3)',   label: 'Pro · 50k'   },
  enterprise: { bg: 'rgba(251,191,36,0.12)',  color: '#fde68a', border: 'rgba(251,191,36,0.3)',   label: 'Enterprise · ∞' },
}

export default function ApiKeys() {
  const [keys,   setKeys]   = useState([])
  const [name,   setName]   = useState('')
  const [plan,   setPlan]   = useState('free')
  const [newKey, setNewKey] = useState(null)
  const [copied, setCopied] = useState(false)
  const [active, setActive] = useState(localStorage.getItem('gr_api_key') ?? '')

  async function load() {
    try { const r = await api.get('/api/v1/keys/'); setKeys(r.data) } catch {}
  }

  useEffect(() => { load() }, [])

  async function create() {
    if (!name.trim()) return
    try {
      const r = await fetch('/api/v1/keys/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), plan }),
      })
      const data = await r.json()
      setNewKey(data.key)
      setName('')
      load()
    } catch {}
  }

  async function revoke(id) {
    if (!confirm('Revoke this key? This cannot be undone.')) return
    await fetch(`/api/v1/keys/${id}`, { method: 'DELETE' })
    load()
  }

  function copy(key) {
    navigator.clipboard.writeText(key)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function activate(key) { setApiKey(key); setActive(key) }

  return (
    <div className="p-8 max-w-2xl space-y-8">
      <div>
        <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>API Keys</h1>
        <p className="text-sm mt-0.5" style={{ color: 'var(--text-3)' }}>Create and manage keys for the GuardRail API</p>
      </div>

      {/* Active key */}
      {active && (
        <div
          className="rounded-xl px-4 py-3 flex items-center gap-3"
          style={{ background: 'var(--accent-dim)', border: '1px solid rgba(139,92,246,0.25)' }}
        >
          <Zap size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
          <div className="flex-1 min-w-0">
            <p className="text-xs" style={{ color: 'var(--text-3)' }}>Active key</p>
            <p className="text-sm font-mono truncate" style={{ color: '#c4b5fd' }}>{active.slice(0, 24)}…</p>
          </div>
        </div>
      )}

      {/* Create */}
      <div className="card p-5 space-y-4">
        <p className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>Create New Key</p>
        <div className="flex gap-2">
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && create()}
            placeholder="Key name (e.g. production)"
            className="input flex-1"
          />
          <select
            value={plan}
            onChange={e => setPlan(e.target.value)}
            className="input"
            style={{ width: 'auto' }}
          >
            <option value="free">Free · 1k</option>
            <option value="pro">Pro · 50k</option>
            <option value="enterprise">Enterprise · ∞</option>
          </select>
          <button onClick={create} disabled={!name.trim()} className="btn-primary" style={{ whiteSpace: 'nowrap' }}>
            <Plus size={14} /> Create
          </button>
        </div>

        {/* New key reveal */}
        {newKey && (
          <div
            className="rounded-xl p-4 space-y-3"
            style={{ background: 'rgba(52,211,153,0.07)', border: '1px solid rgba(52,211,153,0.2)' }}
          >
            <div className="flex items-center gap-2 text-xs" style={{ color: '#6ee7b7' }}>
              <AlertTriangle size={12} />
              Copy this key now — it won't be shown again
            </div>
            <div className="flex items-start justify-between gap-3">
              <code className="text-sm font-mono break-all" style={{ color: '#34d399' }}>{newKey}</code>
              <div className="flex gap-2 shrink-0">
                <button onClick={() => copy(newKey)} className="btn-secondary text-xs py-1.5 px-3">
                  {copied ? <Check size={12} style={{ color: '#34d399' }} /> : <Copy size={12} />}
                  {copied ? 'Copied' : 'Copy'}
                </button>
                <button onClick={() => activate(newKey)} className="btn-primary text-xs py-1.5 px-3">
                  Use key
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Keys list */}
      <div className="space-y-2">
        <p className="label">Your Keys ({keys.length})</p>
        {keys.length === 0 ? (
          <div className="card p-10 text-center">
            <Key size={24} className="mx-auto mb-3" style={{ color: 'var(--text-3)' }} />
            <p className="text-sm" style={{ color: 'var(--text-3)' }}>No keys yet. Create one above.</p>
          </div>
        ) : (
          keys.map(k => {
            const ps = PLAN_STYLE[k.plan] ?? PLAN_STYLE.free
            const usagePct = k.limit < 10_000_000 ? Math.min(100, (k.requests / k.limit) * 100) : 0
            return (
              <div key={k.id} className="card p-4 flex items-center gap-4">
                <div
                  className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
                >
                  <Key size={14} style={{ color: 'var(--text-3)' }} />
                </div>
                <div className="flex-1 min-w-0 space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium truncate" style={{ color: 'var(--text-1)' }}>{k.name}</p>
                    <span
                      className="badge text-xs"
                      style={{ background: ps.bg, color: ps.color, borderColor: ps.border }}
                    >
                      {ps.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <code className="text-xs font-mono" style={{ color: 'var(--text-3)' }}>{k.key_preview}</code>
                    <span className="text-xs" style={{ color: 'var(--text-3)' }}>
                      {k.requests.toLocaleString()} / {k.limit >= 10_000_000 ? '∞' : k.limit.toLocaleString()}
                    </span>
                  </div>
                  {usagePct > 0 && (
                    <div className="h-1 rounded-full overflow-hidden" style={{ background: 'var(--bg-elevated)' }}>
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${usagePct}%`,
                          background: usagePct > 80 ? '#f87171' : usagePct > 50 ? '#fbbf24' : '#8b5cf6',
                        }}
                      />
                    </div>
                  )}
                </div>
                <button
                  onClick={() => revoke(k.id)}
                  className="p-2 rounded-lg transition-all"
                  style={{ color: 'var(--text-3)' }}
                  onMouseEnter={e => { e.currentTarget.style.color = '#f87171'; e.currentTarget.style.background = 'rgba(248,113,113,0.1)' }}
                  onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-3)'; e.currentTarget.style.background = 'transparent' }}
                  title="Revoke key"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
