import React from 'react'
import { NavLink } from 'react-router-dom'

function Navbar() {
  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand">
        <div className="logo-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
        </div>
        <span>StockAI Pro</span>
      </NavLink>
      <div className="navbar-links">
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
      </div>
    </nav>
  )
}

export default Navbar
