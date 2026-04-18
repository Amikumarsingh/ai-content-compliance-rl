import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { Shield, LayoutDashboard, FlaskConical, BarChart2, Key } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Playground from './pages/Playground'
import Analytics from './pages/Analytics'
import ApiKeys from './pages/ApiKeys'

const nav = [
  { to: '/',            icon: LayoutDashboard, label: 'Dashboard'  },
  { to: '/playground',  icon: FlaskConical,    label: 'Playground' },
  { to: '/analytics',   icon: BarChart2,       label: 'Analytics'  },
  { to: '/keys',        icon: Key,             label: 'API Keys'   },
]

export default function App() {
  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="flex items-center gap-2 px-5 py-5 border-b border-gray-800">
          <Shield className="text-brand-500" size={22} />
          <span className="font-bold text-lg tracking-tight">GuardRail</span>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-brand-500/20 text-brand-500'
                    : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="px-5 py-4 border-t border-gray-800 text-xs text-gray-500">
          v1.0.0 · RL-powered
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/"           element={<Dashboard />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="/analytics"  element={<Analytics />} />
          <Route path="/keys"       element={<ApiKeys />} />
        </Routes>
      </main>
    </div>
  )
}
