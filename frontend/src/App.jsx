import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { Shield, LayoutDashboard, FlaskConical, BarChart2, Key, Activity } from 'lucide-react'
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
    <div className="flex h-screen overflow-hidden bg-surface text-slate-100">
      {/* ── Sidebar ── */}
      <aside className="w-60 flex-shrink-0 flex flex-col border-r border-surface-border bg-surface-card">
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 h-16 border-b border-surface-border">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-brand-500/20 border border-brand-500/30">
            <Shield size={16} className="text-brand-400" />
          </div>
          <div>
            <p className="font-bold text-sm tracking-tight text-slate-100">GuardRail</p>
            <p className="text-[10px] text-slate-500 leading-none">AI Moderation Platform</p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          <p className="section-title px-3 mb-3">Navigation</p>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-brand-500/15 text-brand-400 border border-brand-500/20'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-surface-hover border border-transparent'
                }`
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-surface-border">
          <div className="flex items-center gap-2">
            <Activity size={12} className="text-green-400" />
            <span className="text-xs text-slate-500">v1.0.0 · RL-powered</span>
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
