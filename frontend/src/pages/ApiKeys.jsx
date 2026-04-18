import React, { useEffect, useState } from 'react'
import { Plus, Trash2, Copy, Check, Key, AlertCircle } from 'lucide-react'
import { api, setApiKey } from '../hooks/useApi'

const PLAN_BADGE = {
  free:       'bg-slate-500/10 text-slate-400 border-slate-500/20',
  pro:        'bg-brand-500/10 text-brand-400 border-brand-500/20',
  enterprise: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
}

const PLAN_LIMITS = { free: '1k', pro: '50k', enterprise: '∞' }

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

  function activate(key) {
    setApiKey(key)
    setActive(key)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') create()
  }

  return (
    <div className="p-8 max-w-2xl space-y-8">
      <div>
        <h1 className="text-xl font-bold text-slate-100">API Keys</h1>
        <p className="text-sm text-slate-500 mt-0.5">Create and manage keys for the GuardRail API</p>
      </div>

      {/* Active key banner */}
      {active && (
        <div className="card border-brand-500/20 bg-brand-500/5 px-4 py-3 flex items-center gap-3">
          <Key size={14} className="text-brand-400 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-xs text-slate-400">Active key in use</p>
            <p className="text-sm font-mono text-brand-400 truncate">{active.slice(0, 20)}…</p>
          </div>
        </div>
      )}

      {/* Create form */}
      <div className="card p-5 space-y-4">
        <p className="text-sm font-semibold text-slate-200">Create New Key</p>
        <div className="flex gap-2">
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Key name (e.g. production)"
            className="input flex-1"
          />
          <select
            value={plan}
            onChange={e => setPlan(e.target.value)}
            className="input w-auto"
          >
            <option value="free">Free · 1k</option>
            <option value="pro">Pro · 50k</option>
            <option value="enterprise">Enterprise · ∞</option>
          </select>
          <button onClick={create} disabled={!name.trim()} className="btn-primary shrink-0">
            <Plus size={14} /> Create
          </button>
        </div>

        {/* New key reveal */}
        {newKey && (
          <div className="bg-emerald-500/8 border border-emerald-500/20 rounded-lg p-3 space-y-2">
            <div className="flex items-center gap-2 text-xs text-emerald-400">
              <AlertCircle size={12} />
              Copy this key now — it won't be shown again
            </div>
            <div className="flex items-center justify-between gap-3">
              <code className="text-sm font-mono text-emerald-300 break-all">{newKey}</code>
              <div className="flex gap-2 shrink-0">
                <button onClick={() => copy(newKey)} className="btn-ghost text-xs py-1.5">
                  {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
                  {copied ? 'Copied' : 'Copy'}
                </button>
                <button onClick={() => activate(newKey)} className="btn-primary text-xs py-1.5">
                  Use key
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Keys list */}
      <div className="space-y-2">
        <p className="section-title">Your Keys ({keys.length})</p>
        {keys.length === 0 ? (
          <div className="card p-8 text-center">
            <Key size={24} className="mx-auto mb-2 text-slate-700" />
            <p className="text-sm text-slate-500">No keys yet. Create one above.</p>
          </div>
        ) : (
          keys.map(k => (
            <div key={k.id} className="card p-4 flex items-center gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium text-slate-200 truncate">{k.name}</p>
                  <span className={`badge ${PLAN_BADGE[k.plan] ?? PLAN_BADGE.free}`}>
                    {k.plan} · {PLAN_LIMITS[k.plan]}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-1">
                  <code className="text-xs font-mono text-slate-500">{k.key_preview}</code>
                  <span className="text-xs text-slate-600">{k.requests.toLocaleString()} / {k.limit === 10_000_000 ? '∞' : k.limit.toLocaleString()} requests</span>
                </div>
              </div>
              <button
                onClick={() => revoke(k.id)}
                className="p-2 text-slate-600 hover:text-red-400 transition-colors rounded-lg hover:bg-red-500/10"
                title="Revoke key"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
