import React, { useEffect, useState } from 'react'
import { Plus, Trash2, Copy, Check } from 'lucide-react'
import { api, setApiKey } from '../hooks/useApi'

export default function ApiKeys() {
  const [keys,    setKeys]    = useState([])
  const [name,    setName]    = useState('')
  const [plan,    setPlan]    = useState('free')
  const [newKey,  setNewKey]  = useState(null)
  const [copied,  setCopied]  = useState(false)
  const [active,  setActive]  = useState(localStorage.getItem('gr_api_key') || '')

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
        body: JSON.stringify({ name, plan }),
      })
      const data = await r.json()
      setNewKey(data.key)
      setName('')
      load()
    } catch {}
  }

  async function revoke(id) {
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

  return (
    <div className="p-8 max-w-2xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">API Keys</h1>
        <p className="text-gray-400 text-sm mt-1">Create and manage keys for the GuardRail API</p>
      </div>

      {/* Create */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-semibold text-gray-300">Create New Key</h2>
        <div className="flex gap-3">
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Key name (e.g. production)"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-brand-500"
          />
          <select
            value={plan}
            onChange={e => setPlan(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none"
          >
            <option value="free">Free (1k req)</option>
            <option value="pro">Pro (50k req)</option>
            <option value="enterprise">Enterprise (∞)</option>
          </select>
          <button
            onClick={create}
            className="flex items-center gap-2 px-4 py-2 bg-brand-500 hover:bg-brand-600 rounded-lg text-sm font-semibold transition-colors"
          >
            <Plus size={15} /> Create
          </button>
        </div>

        {newKey && (
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3 flex items-center justify-between">
            <div>
              <p className="text-xs text-green-400 mb-1">New key created — copy it now, it won't be shown again</p>
              <code className="text-sm font-mono text-green-300">{newKey}</code>
            </div>
            <div className="flex gap-2">
              <button onClick={() => copy(newKey)} className="p-2 hover:text-green-300 transition-colors">
                {copied ? <Check size={15} /> : <Copy size={15} />}
              </button>
              <button onClick={() => activate(newKey)} className="text-xs px-3 py-1 bg-green-500/20 hover:bg-green-500/30 rounded-lg transition-colors text-green-400">
                Use this key
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Active key indicator */}
      {active && (
        <div className="bg-brand-500/10 border border-brand-500/30 rounded-xl px-4 py-3 text-sm text-brand-500">
          Active key: <code className="font-mono">{active.slice(0, 14)}…</code>
        </div>
      )}

      {/* List */}
      <div className="space-y-3">
        {keys.map(k => (
          <div key={k.id} className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-200">{k.name}</p>
              <p className="text-xs text-gray-500 mt-0.5">
                <code className="font-mono">{k.key_preview}</code> · {k.plan} · {k.requests}/{k.limit} requests
              </p>
            </div>
            <button onClick={() => revoke(k.id)} className="p-2 text-gray-600 hover:text-red-400 transition-colors">
              <Trash2 size={15} />
            </button>
          </div>
        ))}
        {keys.length === 0 && (
          <p className="text-gray-600 text-sm text-center py-8">No keys yet. Create one above.</p>
        )}
      </div>
    </div>
  )
}
