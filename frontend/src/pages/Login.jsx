import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'

function Login() {
  const [isLogin, setIsLogin] = useState(true)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState('user') // Default role for testing

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)

  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(null)

    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup'
    const body = isLogin
      ? { username, password }
      : { username, email, password, role }

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.error || 'Something went wrong')
      }

      // Save token & metadata to local storage
      localStorage.setItem('token', data.access_token)
      localStorage.setItem('username', data.username)
      localStorage.setItem('role', data.role)

      setSuccess(isLogin ? 'Login successful!' : 'Signup successful! Logging you in...')

      // Fire global storage event to update Navbar state immediately
      window.dispatchEvent(new Event('storage'))

      setTimeout(() => {
        navigate('/')
      }, 1000)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: '420px', margin: '80px auto', padding: '10px' }}>
      <div className="card" style={{ padding: '36px', position: 'relative' }}>
        <div style={{ display: 'flex', gap: '20px', marginBottom: '30px', borderBottom: '1px solid var(--glass-border)', paddingBottom: '12px' }}>
          <button
            onClick={() => { setIsLogin(true); setError(null); }}
            style={{
              background: 'none',
              border: 'none',
              color: isLogin ? 'var(--accent-blue)' : 'var(--text-secondary)',
              fontSize: '1.05rem',
              fontWeight: 800,
              cursor: 'pointer',
              padding: '0 4px 8px 4px',
              borderBottom: isLogin ? '2px solid var(--accent-blue)' : 'none',
              transition: 'all 0.2s'
            }}
          >
            Log In
          </button>
          <button
            onClick={() => { setIsLogin(false); setError(null); }}
            style={{
              background: 'none',
              border: 'none',
              color: !isLogin ? 'var(--accent-blue)' : 'var(--text-secondary)',
              fontSize: '1.05rem',
              fontWeight: 800,
              cursor: 'pointer',
              padding: '0 4px 8px 4px',
              borderBottom: !isLogin ? '2px solid var(--accent-blue)' : 'none',
              transition: 'all 0.2s'
            }}
          >
            Sign Up
          </button>
        </div>

        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, marginBottom: '20px' }}>
          {isLogin ? 'Welcome Back' : 'Create Account'}
        </h2>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>
              USERNAME
            </label>
            <input
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid var(--glass-border)',
                borderRadius: '8px',
                padding: '11px 16px',
                color: 'white',
                outline: 'none',
                fontSize: '0.92rem'
              }}
            />
          </div>

          {!isLogin && (
            <>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>
                  EMAIL ADDRESS
                </label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@company.com"
                  style={{
                    width: '100%',
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: '8px',
                    padding: '11px 16px',
                    color: 'white',
                    outline: 'none',
                    fontSize: '0.92rem'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>
                  ACCOUNT ROLE
                </label>
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(30,30,45,1)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: '8px',
                    padding: '11px 16px',
                    color: 'white',
                    outline: 'none',
                    fontSize: '0.92rem'
                  }}
                >
                  <option value="user">Regular User</option>
                  <option value="admin">Administrator</option>
                </select>
              </div>
            </>
          )}

          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>
              PASSWORD
            </label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid var(--glass-border)',
                borderRadius: '8px',
                padding: '11px 16px',
                color: 'white',
                outline: 'none',
                fontSize: '0.92rem'
              }}
            />
          </div>

          {error && (
            <div style={{ color: 'var(--accent-red)', fontSize: '0.82rem', fontWeight: 600, padding: '10px', borderRadius: '6px', background: 'rgba(255, 71, 87, 0.08)', border: '1px solid rgba(255, 71, 87, 0.15)' }}>
              ⚠️ {error}
            </div>
          )}

          {success && (
            <div style={{ color: 'var(--accent-green)', fontSize: '0.82rem', fontWeight: 600, padding: '10px', borderRadius: '6px', background: 'rgba(0, 212, 170, 0.08)', border: '1px solid rgba(0, 212, 170, 0.15)' }}>
              ✓ {success}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary"
            style={{ width: '100%', padding: '12px', borderRadius: '8px', justifyContent: 'center', marginTop: '10px', fontWeight: 700 }}
          >
            {loading ? 'Processing...' : isLogin ? 'Sign In' : 'Register Account'}
          </button>
        </form>
        
        {isLogin && (
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textAlign: 'center', marginTop: '20px' }}>
            💡 Demo Account Check:<br />
            Username: <strong style={{ color: 'white' }}>admin</strong> | Password: <strong style={{ color: 'white' }}>adminpassword</strong>
          </p>
        )}
      </div>
    </div>
  )
}

export default Login
