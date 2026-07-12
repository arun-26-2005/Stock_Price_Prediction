import React from 'react'

function NewsCard({ title, publisher, sentiment, confidence, link, publishTime }) {
  const getSentimentClass = () => {
    if (sentiment === 'positive') return 'badge-success'
    if (sentiment === 'negative') return 'badge-danger'
    return 'badge-neutral'
  };

  const getBarColor = () => {
    if (sentiment === 'positive') return 'var(--accent-green)'
    if (sentiment === 'negative') return 'var(--accent-red)'
    return 'var(--text-secondary)'
  };

  const getSentimentEmoji = () => {
    if (sentiment === 'positive') return '📈'
    if (sentiment === 'negative') return '📉'
    return '➖'
  };

  const accentBorder = sentiment === 'positive'
    ? 'var(--accent-green)'
    : sentiment === 'negative'
      ? 'var(--accent-red)'
      : 'var(--border-color)';

  return (
    <div
      className={`card news-card news-card-${sentiment || 'neutral'}`}
      style={{ borderLeft: `3px solid ${accentBorder}` }}
    >
      <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
        <span style={{ fontSize: '1.4rem', lineHeight: '1' }}>{getSentimentEmoji()}</span>
        <h3 className="news-card-title">{title}</h3>
      </div>
      <div className="news-card-footer">
        <div className="news-card-meta">
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ opacity: 0.5 }}>📰</span> {publisher}
          </span>
          <span className={`stock-card-badge ${getSentimentClass()}`}>
            {sentiment}
          </span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              AI Confidence
            </span>
            <span style={{ fontSize: '0.72rem', color: getBarColor(), fontWeight: 700 }}>
              {Math.round(confidence * 100)}%
            </span>
          </div>
          <div className="confidence-bar-container">
            <div
              className="confidence-bar"
              style={{
                width: `${confidence * 100}%`,
                backgroundColor: getBarColor(),
                boxShadow: `0 0 8px ${getBarColor()}`
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default NewsCard
