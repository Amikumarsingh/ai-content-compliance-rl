import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { Shield, LayoutDashboard, FlaskConical, BarChart2, Key, Zap } from 'lucide-react'
import Dashboard  from './pages/Dashboard'
import Playground from './pages/Playground'
import Analytics  from './pages/Analytics'
import ApiKeys    from './pages/ApiKeys'

const NAV = [
  { to: '/',            icon: LayoutDashboard, label: 'Dashboard'  },
  { to: '/playground',  icon: FlaskConical,    label: 'Playground' },
  { to: '/analytics',   icon: BarChart2,       label: 'Analytics'  },
  { to: '/keys',        icon: Key,             label: 'API Keys'   },
]

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>

      {/* ── Sidebar ── */}
      <aside
        className="w-60 flex-shrink-0 flex flex-col"
        style={{
          background: 'var(--bg-card)',
          borderRight: '1px solid var(--border)',
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 h-16" style={{ borderBottom: '1px solid var(--border)' }}>
          <div
            className="flex items-center justify-center w-8 h-8 rounded-lg"
            style={{ background: 'var(--accent-dim)', border: '1px solid rgba(139,92,246,0.25)' }}
          >
            <Shield size={15} style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <p className="font-bold text-sm" style={{ color: 'var(--text-1)' }}>GuardRail</p>
            <p style={{ fontSize: 10, color: 'var(--text-3)', lineHeight: 1.2 }}>AI Moderation</p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-5 space-y-0.5">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => isActive ? 'nav-item-active' : 'nav-item'}
              style={({ isActive }) => isActive
                ? { display:'flex', alignItems:'center', gap:10, padding:'8px 12px', borderRadius:8,
                    background:'var(--accent-dim)', color:'#c4b5fd',
                    border:'1px solid rgba(139,92,246,0.2)', fontSize:13, fontWeight:500 }
                : { display:'flex', alignItems:'center', gap:10, padding:'8px 12px', borderRadius:8,
                    color:'var(--text-2)', border:'1px solid transparent', fontSize:13, fontWeight:500,
                    transition:'all 0.15s' }
              }
              onMouseEnter={e => { if (!e.currentTarget.classList.contains('nav-item-active')) { e.currentTarget.style.background = 'var(--bg-hover)'; e.currentTarget.style.color = 'var(--text-1)' }}}
              onMouseLeave={e => { if (!e.currentTarget.classList.contains('nav-item-active')) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-2)' }}}
            >
              <Icon size={14} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span style={{ fontSize: 11, color: 'var(--text-3)' }}>v1.0.0 · RL-powered</span>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/"           element={<Dashboard />}  />
          <Route path="/playground" element={<Playground />} />
          <Route path="/analytics"  element={<Analytics />}  />
          <Route path="/keys"       element={<ApiKeys />}    />
        </Routes>
      </main>
    </div>
  )
}
