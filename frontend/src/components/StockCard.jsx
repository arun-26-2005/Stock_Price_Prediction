import React from 'react'

const STOCK_ICONS = {
  TCS: '💻',
  RELIANCE: '⛽',
  INFY: '🔷',
  NIFTY: '📈',
}

const STOCK_DESCRIPTIONS = {
  TCS: 'Tata Consultancy Services — India\'s largest IT services company',
  RELIANCE: 'Reliance Industries — Diversified conglomerate (Oil, Telecom, Retail)',
  INFY: 'Infosys — Global IT consulting & outsourcing leader',
  NIFTY: 'Nifty 50 Index — Benchmark index of 50 Indian blue-chip stocks',
}

function StockCard({ name, symbol, hasCheckpoint, onClick }) {
  return (
    <div className={`card stock-card ${hasCheckpoint ? 'active' : ''}`} onClick={onClick}>
      <div className="stock-card-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="stock-card-icon">{STOCK_ICONS[name] || '📊'}</span>
          <span className="stock-card-symbol">{symbol}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {hasCheckpoint && <span className="pulse-dot" />}
          <span className={`stock-card-badge ${hasCheckpoint ? 'badge-success' : 'badge-neutral'}`}>
            {hasCheckpoint ? '● Trained' : '○ Ready'}
          </span>
        </div>
      </div>
      <div style={{ marginTop: 'auto' }}>
        <h2 className="stock-card-name">{name}</h2>
        <p className="stock-card-desc">
          {STOCK_DESCRIPTIONS[name] || (hasCheckpoint ? 'Full ML pipeline enabled' : 'Click to run model training')}
        </p>
      </div>
      <div className="stock-card-action">
        <span>{hasCheckpoint ? 'View Predictions →' : 'Train Model →'}</span>
      </div>
    </div>
  )
}

export default StockCard
