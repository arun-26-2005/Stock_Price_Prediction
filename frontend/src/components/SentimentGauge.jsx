import React from 'react'

function SentimentGauge({ score }) {
  // Map score [-1, 1] to angle [0, 180] degrees (left to right)
  const angle = score * 90;

  const getSentimentLabel = () => {
    if (score >= 0.15) return 'Bullish'
    if (score <= -0.15) return 'Bearish'
    return 'Neutral'
  };

  const getSentimentColor = () => {
    if (score >= 0.15) return 'var(--accent-green)'
    if (score <= -0.15) return 'var(--accent-red)'
    return 'var(--accent-blue)'
  };

  const getSentimentGlow = () => {
    if (score >= 0.15) return 'var(--accent-green-glow)'
    if (score <= -0.15) return 'var(--accent-red-glow)'
    return 'var(--accent-blue-glow)'
  };

  const getSentimentEmoji = () => {
    if (score >= 0.15) return '🟢'
    if (score <= -0.15) return '🔴'
    return '🔵'
  };

  return (
    <div className="sentiment-gauge-container">
      <svg width="220" height="130" viewBox="0 0 200 120">
        <defs>
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ef4444" />
            <stop offset="25%" stopColor="#f59e0b" />
            <stop offset="50%" stopColor="#a78bfa" />
            <stop offset="75%" stopColor="#34d399" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
          <filter id="needleGlow">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background Arc (Track) */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="rgba(255,255,255,0.04)"
          strokeWidth="14"
          strokeLinecap="round"
        />

        {/* Colored Gauge Arc */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth="14"
          strokeLinecap="round"
          style={{ filter: 'drop-shadow(0 0 6px rgba(59,130,246,0.3))' }}
        />

        {/* Tick marks */}
        {[-60, -30, 0, 30, 60].map((deg, i) => {
          const rad = (deg - 90) * Math.PI / 180;
          const x1 = 100 + 74 * Math.cos(rad);
          const y1 = 100 + 74 * Math.sin(rad);
          const x2 = 100 + 68 * Math.cos(rad);
          const y2 = 100 + 68 * Math.sin(rad);
          return (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" strokeLinecap="round" />
          );
        })}

        {/* Center Hub — outer ring */}
        <circle cx="100" cy="100" r="12" fill="rgba(15,15,21,0.9)" stroke="rgba(255,255,255,0.1)" strokeWidth="1.5" />

        {/* Center Hub — glowing core */}
        <circle cx="100" cy="100" r="5" fill={getSentimentColor()} filter="url(#glow)" opacity="0.9" />

        {/* Needle */}
        <g transform={`rotate(${angle}, 100, 100)`} style={{ transition: 'transform 0.8s cubic-bezier(0.25, 1, 0.5, 1)' }}>
          <line
            x1="100"
            y1="105"
            x2="100"
            y2="28"
            stroke={getSentimentColor()}
            strokeWidth="2.5"
            strokeLinecap="round"
            filter="url(#needleGlow)"
          />
          {/* Needle tip */}
          <circle cx="100" cy="28" r="3" fill={getSentimentColor()} filter="url(#glow)" />
        </g>

        {/* Labels */}
        <text x="25" y="118" fill="rgba(255,255,255,0.35)" fontSize="8" fontWeight="600" textAnchor="middle" fontFamily="var(--font-family)">BEARISH</text>
        <text x="100" y="15" fill="rgba(255,255,255,0.35)" fontSize="8" fontWeight="600" textAnchor="middle" fontFamily="var(--font-family)">NEUTRAL</text>
        <text x="175" y="118" fill="rgba(255,255,255,0.35)" fontSize="8" fontWeight="600" textAnchor="middle" fontFamily="var(--font-family)">BULLISH</text>
      </svg>

      <div
        className="sentiment-gauge-label"
        style={{
          color: getSentimentColor(),
          textShadow: `0 0 20px ${getSentimentGlow()}`
        }}
      >
        {getSentimentEmoji()} {score >= 0 ? '+' : ''}{score.toFixed(2)} — {getSentimentLabel()}
      </div>
    </div>
  )
}

export default SentimentGauge
