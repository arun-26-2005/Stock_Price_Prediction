import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Plot from 'react-plotly.js'
import LoadingSpinner from '../components/LoadingSpinner'

function AdminDashboard() {
  const [metrics, setMetrics] = useState(null)
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const navigate = useNavigate()
  const token = localStorage.getItem('token')
  const role = localStorage.getItem('role')

  useEffect(() => {
    // Redirect if not logged in or not admin
    if (!token || role !== 'admin') {
      navigate('/')
      return
    }

    const headers = { 'Authorization': `Bearer ${token}` }

    Promise.all([
      fetch('/api/admin/metrics', { headers }).then(r => {
        if (!r.ok) throw new Error('Failed to load admin metrics')
        return r.json()
      }),
      fetch('/api/admin/logs', { headers }).then(r => {
        if (!r.ok) throw new Error('Failed to load user search logs')
        return r.json()
      })
    ])
      .then(([metricsData, logsData]) => {
        setMetrics(metricsData)
        setLogs(logsData.logs || [])
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [token, role, navigate])

  if (loading) return <LoadingSpinner text="Loading Admin monitoring console..." />

  if (error) {
    return (
      <div className="container" style={{ textAlign: 'center', marginTop: '60px' }}>
        <h2 className="section-title" style={{ color: 'var(--accent-red)' }}>Console Error</h2>
        <p style={{ color: 'var(--text-secondary)' }}>{error}</p>
        <button className="btn btn-primary" onClick={() => window.location.reload()} style={{ marginTop: '20px' }}>
          Retry Loading
        </button>
      </div>
    )
  }

  // Prep popular searches chart data
  const chartLabels = metrics?.popular_searches?.map(item => item.symbol) || []
  const chartValues = metrics?.popular_searches?.map(item => item.count) || []

  return (
    <div className="container" style={{ paddingBottom: '80px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        <div>
          <div className="hero-badge" style={{ marginBottom: '8px' }}>
            <span className="pulse-dot" style={{ backgroundColor: 'var(--accent-red)' }} />
            SYSTEM AUDIT
          </div>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 900, background: 'linear-gradient(135deg, #fff 0%, #a5a5cc 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Admin Monitoring Console
          </h1>
        </div>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        <div className="card" style={{ padding: '24px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Total Registered Users
          </span>
          <h2 style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--accent-blue)', margin: '10px 0' }}>
            {metrics?.total_users || 0}
          </h2>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Database instances active
          </span>
        </div>

        <div className="card" style={{ padding: '24px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Total API Queries Logged
          </span>
          <h2 style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--accent-green)', margin: '10px 0' }}>
            {metrics?.total_searches || 0}
          </h2>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Stock searches recorded
          </span>
        </div>

        <div className="card" style={{ padding: '24px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Database Engine
          </span>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 900, color: '#ffb300', margin: '12px 0' }}>
            SQLite v3
          </h2>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Saved at: data/platform.db
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', alignItems: 'start' }}>
        {/* Popular Searches Chart */}
        <div className="card" style={{ padding: '28px' }}>
          <h3 className="card-title" style={{ fontSize: '1rem', fontWeight: 800, marginBottom: '20px' }}>
            📊 Popular Stock Workspaces
          </h3>
          {chartLabels.length > 0 ? (
            <Plot
              data={[
                {
                  x: chartLabels,
                  y: chartValues,
                  type: 'bar',
                  marker: {
                    color: 'rgba(79, 140, 255, 0.65)',
                    line: {
                      color: 'var(--accent-blue)',
                      width: 1.5
                    }
                  }
                }
              ]}
              layout={{
                autosize: true,
                height: 300,
                margin: { l: 40, r: 20, t: 10, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#a5a5cc', family: 'Inter, sans-serif', size: 10 },
                xaxis: {
                  gridcolor: 'rgba(255,255,255,0.03)',
                  tickfont: { size: 9 }
                },
                yaxis: {
                  gridcolor: 'rgba(255,255,255,0.03)',
                  dtick: 1
                }
              }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', textAlign: 'center', margin: '40px 0' }}>
              No search logs recorded yet.
            </p>
          )}
        </div>

        {/* Audit Logs Table */}
        <div className="card" style={{ padding: '28px' }}>
          <h3 className="card-title" style={{ fontSize: '1rem', fontWeight: 800, marginBottom: '20px' }}>
            📋 User Activity & Audit Logs
          </h3>
          <div style={{ maxHeight: '300px', overflowY: 'auto', paddingRight: '8px' }}>
            {logs.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--glass-border)', color: 'var(--text-secondary)', fontWeight: 700 }}>
                    <th style={{ padding: '8px 4px' }}>USER</th>
                    <th style={{ padding: '8px 4px' }}>SYMBOL</th>
                    <th style={{ padding: '8px 4px' }}>TIMESTAMP</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log) => (
                    <tr key={log.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)', color: 'white' }}>
                      <td style={{ padding: '10px 4px', fontWeight: 600 }}>{log.username}</td>
                      <td style={{ padding: '10px 4px' }}>
                        <span className="hero-badge" style={{ padding: '2px 8px', fontSize: '0.7rem', backgroundColor: 'rgba(79, 140, 255, 0.08)', color: 'var(--accent-blue)', border: '1px solid rgba(79, 140, 255, 0.15)' }}>
                          {log.symbol}
                        </span>
                      </td>
                      <td style={{ padding: '10px 4px', color: 'var(--text-secondary)', fontSize: '0.78rem' }}>{log.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', textAlign: 'center', margin: '40px 0' }}>
                No active audit logs.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default AdminDashboard
