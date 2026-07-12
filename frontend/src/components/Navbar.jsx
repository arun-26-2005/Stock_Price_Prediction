import React, { useState, useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'

function Navbar() {
  const [username, setUsername] = useState(localStorage.getItem('username'))
  const [role, setRole] = useState(localStorage.getItem('role'))
  const navigate = useNavigate()

  useEffect(() => {
    const handleStorageChange = () => {
      setUsername(localStorage.getItem('username'))
      setRole(localStorage.getItem('role'))
    }
    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [])

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    localStorage.removeItem('role')
    setUsername(null)
    setRole(null)
    // Dispatch storage event to notify other components
    window.dispatchEvent(new Event('storage'))
    navigate('/')
  }

  return (
    <nav className="navbar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <NavLink to="/" className="navbar-brand" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div className="logo-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
        </div>
        <span>StockAI Pro</span>
      </NavLink>
      
      <div className="navbar-links" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <NavLink
          to="/"
          end
          className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
        >
          <span className="navbar-link-icon">🏠</span>
          Home
        </NavLink>
        <NavLink
          to="/models"
          className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
        >
          <span className="navbar-link-icon">🔬</span>
          Models
        </NavLink>
        <NavLink
          to="/architecture"
          className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
        >
          <span className="navbar-link-icon">🧠</span>
          Architecture
        </NavLink>
        
        {role === 'admin' && (
          <NavLink
            to="/admin"
            className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
            style={{ color: 'var(--accent-red)' }}
          >
            <span className="navbar-link-icon">⚙️</span>
            Admin
          </NavLink>
        )}

        {username ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderLeft: '1px solid var(--glass-border)', paddingLeft: '20px', marginLeft: '10px' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
              👤 {username}
            </span>
            <button
              onClick={handleLogout}
              style={{
                background: 'rgba(255, 71, 87, 0.08)',
                border: '1px solid rgba(255, 71, 87, 0.15)',
                color: 'var(--accent-red)',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '0.78rem',
                fontWeight: 700,
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => {
                e.target.style.background = 'rgba(255, 71, 87, 0.15)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'rgba(255, 71, 87, 0.08)';
              }}
            >
              Log Out
            </button>
          </div>
        ) : (
          <NavLink
            to="/login"
            className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
            style={{
              background: 'rgba(79, 140, 255, 0.08)',
              border: '1px solid rgba(79, 140, 255, 0.15)',
              color: 'var(--accent-blue)',
              padding: '6px 14px',
              borderRadius: '6px',
              fontWeight: 700,
              marginLeft: '10px'
            }}
          >
            Sign In
          </NavLink>
        )}
      </div>
    </nav>
  )
}

export default Navbar
