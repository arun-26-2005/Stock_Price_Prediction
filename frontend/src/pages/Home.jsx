import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import StockCard from '../components/StockCard'
import LoadingSpinner from '../components/LoadingSpinner'

const FEATURE_CARDS = [
  {
    icon: '🧠',
    color: 'var(--accent-blue)',
    title: 'Hybrid Deep Learning',
    desc: 'BiLSTM captures past & future context, Transformers find long-range attention patterns, and TCN extracts local temporal features — all fused into one prediction.',
  },
  {
    icon: '📰',
    color: 'var(--accent-green)',
    title: 'Live News Sentiment',
    desc: 'FinBERT reads today\'s financial headlines and shifts the forecast up or down based on real-time market mood — no stale data.',
  },
  {
    icon: '🌍',
    color: 'var(--accent-yellow)',
    title: 'Global Macro Signals',
    desc: 'Gold, Crude Oil, USD/INR, S&P 500, VIX, and NIFTY 50 are all woven into every prediction to capture worldwide economic drivers.',
  },
  {
    icon: '📊',
    color: '#a78bfa',
    title: 'Walk-Forward Backtester',
    desc: 'Simulates real trading with fees, slippage thresholds, and position sizing. Measures Sharpe ratio, drawdown, and win rate on unseen data.',
  },
  {
    icon: '🤖',
    color: '#f472b6',
    title: 'Personalized Advisor',
    desc: 'Tell us your portfolio position and get a plain-English Buy / Sell / Hold recommendation powered by tomorrow\'s AI forecast.',
  },
  {
    icon: '📈',
    color: '#34d399',
    title: 'Interactive Charts',
    desc: 'Plotly-powered candlestick charts with toggleable overlays — Moving Averages, Bollinger Bands, RSI, MACD — all in dark mode.',
  },
]

function Home() {
  const [stocks, setStocks] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    fetch('/api/stocks')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch stocks')
        return res.json()
      })
      .then((data) => {
        setStocks(data.stocks || [])
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const handleStockClick = (stock) => {
    navigate(`/dashboard/${stock.name.toLowerCase()}`)
  }

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    if (!searchQuery.trim()) return
    
    const ticker = searchQuery.trim().toUpperCase()
    navigate(`/dashboard/${ticker.toLowerCase()}`)
  }

  if (loading) return <LoadingSpinner text="Connecting to FastAPI backend..." />

  if (error) {
    return (
      <div className="error-container">
        <div style={{ fontSize: '3rem', marginBottom: '10px' }}>⚠️</div>
        <h2 className="error-title">Backend Connection Failed</h2>
        <p className="error-text">Please ensure the FastAPI server is running on port 8000.</p>
        <button className="btn btn-primary" onClick={() => window.location.reload()}>
          🔄 Retry Connection
        </button>
      </div>
    )
  }

  return (
    <div>
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-badge">
          <span className="pulse-dot" style={{ width: '6px', height: '6px' }} />
          Live AI Engine
        </div>
        <h1 className="hero-title">
          AI-Powered Stock<br />
          <span className="hero-title-accent">Price Prediction</span>
        </h1>
        <p className="hero-subtitle">
          An advanced forecasting platform that combines deep learning, real-time news sentiment analysis, and global macroeconomic signals to predict tomorrow's stock price.
        </p>
        <div className="hero-stats">
          <div className="hero-stat">
            <span className="hero-stat-value">6</span>
            <span className="hero-stat-label">Model Architectures</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">15+</span>
            <span className="hero-stat-label">Technical Indicators</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">6</span>
            <span className="hero-stat-label">Macro Features</span>
          </div>
        </div>
      </div>

      {/* Search / Add Stock Section */}
      <div className="card" style={{ maxWidth: '600px', margin: '0 auto 50px auto', padding: '24px', textAlign: 'center' }}>
        <h3 className="card-title" style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '8px' }}>
          🔍 Sync Any Indian Stock (NSE)
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '20px', lineHeight: '1.4' }}>
          Enter any NSE symbol (e.g. <strong>TATAMOTORS</strong>, <strong>HDFCBANK</strong>, <strong>SBIN</strong>, <strong>ITC</strong>). The platform will dynamically load the sector-grouped weights and compile indicators in-memory.
        </p>

        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '10px' }}>
          <input
            type="text"
            placeholder="Type NSE symbol (e.g. TATAMOTORS)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              flex: 1,
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid var(--glass-border)',
              borderRadius: '8px',
              padding: '10px 16px',
              color: 'white',
              outline: 'none',
              fontSize: '0.92rem'
            }}
          />
          <button
            type="submit"
            className="btn btn-primary"
            style={{ padding: '0 20px', borderRadius: '8px', minWidth: '110px', justifyContent: 'center' }}
          >
            Open Dashboard
          </button>
        </form>
      </div>

      {/* Stock Cards */}
      <h2 className="section-title">Select Stock Workspace</h2>
      <div className="stock-grid">
        {stocks.map((stock, idx) => (
          <StockCard
            key={stock.name}
            name={stock.name}
            symbol={stock.symbol}
            hasCheckpoint={stock.has_checkpoint}
            onClick={() => handleStockClick(stock)}
          />
        ))}
      </div>

      {/* Features Grid */}
      <h2 className="section-title" style={{ marginTop: '60px' }}>Platform Capabilities</h2>
      <div className="features-grid">
        {FEATURE_CARDS.map((feat, idx) => (
          <div
            className="card feature-card"
            key={idx}
            style={{ '--accent': feat.color, animationDelay: `${idx * 0.08}s` }}
          >
            <div className="feature-card-icon" style={{ background: `${feat.color}15`, color: feat.color }}>
              {feat.icon}
            </div>
            <h3 className="feature-card-title">{feat.title}</h3>
            <p className="feature-card-desc">{feat.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default Home
