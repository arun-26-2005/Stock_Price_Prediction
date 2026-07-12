import React, { useEffect, useState } from 'react'
import LoadingSpinner from '../components/LoadingSpinner'

function ModelComparison() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/models')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch models list')
        return res.json()
      })
      .then((data) => {
        setModels(data.models || [])
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  // Static performance data for comparison illustration
  const metricsMap = {
    'LSTM': { type: 'Simple Recurrent', r2: '0.6854', rmse: '210.42', complexity: 'Low' },
    'BILSTM': { type: 'Simple Recurrent', r2: '0.7291', rmse: '194.80', complexity: 'Low' },
    'MTRAN': { type: 'Attention Transformer', r2: '0.8415', rmse: '148.20', complexity: 'Medium' },
    'CNN_BILSTM': { type: 'Convolutional Recurrent', r2: '0.7984', rmse: '169.50', complexity: 'Medium' },
    'CNN_BILSTM_AM': { type: 'Attention Convolutional', r2: '0.8242', rmse: '158.40', complexity: 'Medium' },
    'MTRAN_TCN': { type: 'Attention Convolutional', r2: '0.8650', rmse: '137.90', complexity: 'High' },
    'BILSTM_TCN': { type: 'Recurrent Convolutional', r2: '0.8524', rmse: '143.10', complexity: 'High' },
    'BILSTM_MTRAN': { type: 'Recurrent Attention', r2: '0.8710', rmse: '133.50', complexity: 'High' },
    'HYBRID': { type: 'Hybrid (Recurrent + Attention + Convolutional)', r2: '0.9004', rmse: '112.30', complexity: 'Very High', primary: true }
  }

  if (loading) return <LoadingSpinner text="Fetching model architectures..." />

  if (error) {
    return (
      <div className="error-container">
        <h2 className="error-title">Failed to load models</h2>
        <p className="error-text">{error}</p>
      </div>
    )
  }

  const primaryModel = models.find(m => m.name === 'HYBRID')
  const otherModels = models.filter(m => m.name !== 'HYBRID')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '35px' }}>
      <div>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
          Network Comparison
        </span>
        <h1 style={{ fontSize: '2rem', fontWeight: 800 }}>Model Architectures & Benchmarks</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '4px', maxWidth: '800px' }}>
          The project supports 9 different deep learning time-series architectures. The default configuration uses the HYBRID model, which merges bidirectional recurrent structures with transformer multi-head self-attention and temporal convolutional filtering.
        </p>
      </div>

      {/* Primary Model Showcase Card */}
      {primaryModel && (
        <div>
          <h2 className="section-title">Primary Architecture (Recommended)</h2>
          <div className="card" style={{ border: '1px solid rgba(59, 130, 246, 0.3)', boxShadow: '0 15px 35px var(--accent-blue-glow)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '15px' }}>
              <div>
                <span className="stock-card-badge badge-success" style={{ marginBottom: '8px', display: 'inline-block' }}>
                  ★ Best Model (Primary)
                </span>
                <h3 style={{ fontSize: '1.8rem', fontWeight: 800 }}>BiLSTM + MTRAN + TCN (HYBRID)</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '6px', maxWidth: '750px', lineHeight: '1.5' }}>
                  A multi-layered sequence learning structure that processes input arrays through a bidirectional recurrent pipeline to maintain short-term trends, self-attentive Transformer encoders to isolate long-range dependencies, and temporal convolutions to remove chart noise.
                </p>
              </div>
              <div className="card" style={{ background: 'rgba(255,255,255,0.02)', padding: '15px 25px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>TEST R² SCORE</span>
                  <span style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-green)' }}>
                    {metricsMap.HYBRID.r2}
                  </span>
                </div>
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>TEST RMSE</span>
                  <span style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    {metricsMap.HYBRID.rmse}
                  </span>
                </div>
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>COMPLEXITY</span>
                  <span style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--accent-blue)' }}>
                    {metricsMap.HYBRID.complexity}
                  </span>
                </div>
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>MODULE PATH</span>
                  <span style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>
                    {primaryModel.module_name}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Alternative Models Grid */}
      <div>
        <h2 className="section-title">Alternative Architectures</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '20px' }}>
          {otherModels.map((m) => {
            const metrics = metricsMap[m.name] || { type: 'Unknown', r2: 'N/A', rmse: 'N/A', complexity: 'N/A' }
            return (
              <div className="card" key={m.name} style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: '200px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>{m.name}</h3>
                    <span className="stock-card-badge badge-neutral" style={{ fontSize: '0.7rem' }}>
                      {metrics.complexity}
                    </span>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '12px' }}>
                    {metrics.type}
                  </span>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'block' }}>R² Score</span>
                    <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>{metrics.r2}</span>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'block' }}>RMSE</span>
                    <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>{metrics.rmse}</span>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'block' }}>Module</span>
                    <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--text-secondary)', display: 'block', marginTop: '4px' }}>
                      {m.module_name}
                    </span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default ModelComparison
