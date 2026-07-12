import React from 'react'

function Sparkline({ data, color }) {
  if (!data || data.length < 2) return null;
  const width = 100;
  const height = 28;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min === 0 ? 1 : max - min;
  
  const points = data.map((val, idx) => {
    const x = (idx / (data.length - 1)) * width;
    const y = height - 2 - ((val - min) / range) * (height - 4);
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} style={{ overflow: 'visible', opacity: 0.85 }}>
      <polyline
        fill="none"
        stroke={color || 'var(--accent-blue)'}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

function KPICard({ title, value, subtitle, trend, icon, showExplainer, explainerText, sparklineData, sparklineColor }) {
  const getTrendClass = () => {
    if (trend === 'up') return 'badge-success'
    if (trend === 'down') return 'badge-danger'
    if (trend === 'warning') return 'badge-warning'
    return 'badge-neutral'
  };

  const getTrendIcon = () => {
    if (trend === 'up') return '▲'
    if (trend === 'down') return '▼'
    return ''
  };

  return (
    <div className="card kpi-card">
      <span className="kpi-title">{title}</span>
      {showExplainer && explainerText && (
        <div style={{ fontSize: '0.72rem', color: 'var(--accent-blue)', background: 'var(--accent-blue-glow)', padding: '6px 10px', borderRadius: '6px', margin: '4px 0', lineHeight: '1.3', border: '1px solid rgba(59,130,246,0.1)' }}>
          💡 {explainerText}
        </div>
      )}
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: showExplainer && explainerText ? '12px' : '6px' }}>
        <div className="kpi-value" style={{ margin: 0, flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
          <span style={{ fontSize: '1.5rem', fontWeight: 800 }}>{value}</span>
          {trend && (
            <span className={`stock-card-badge ${getTrendClass()}`} style={{ fontSize: '0.7rem', padding: '2px 8px' }}>
              {getTrendIcon()} {subtitle}
            </span>
          )}
        </div>
        {sparklineData && (
          <div style={{ marginLeft: '10px', marginBottom: '2px' }}>
            <Sparkline data={sparklineData} color={sparklineColor} />
          </div>
        )}
      </div>
      {!trend && subtitle && <span className="kpi-subtitle" style={{ marginTop: '6px' }}>{subtitle}</span>}
    </div>
  )
}

export default KPICard
