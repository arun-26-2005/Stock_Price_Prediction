import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import LoadingSpinner from '../components/LoadingSpinner'

function MacroHeatmap() {
  const { stock } = useParams()
  const stockName = stock.toUpperCase()

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    
    fetch(`/api/macro/${stockName}`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch correlation matrix')
        return res.json()
      })
      .then((resData) => {
        if (resData.error) throw new Error(resData.error)
        setData(resData)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [stockName])

  if (loading) return <LoadingSpinner text={`Computing macroeconomic correlations for ${stockName}...`} />

  if (error) {
    return (
      <div className="error-container">
        <h2 className="error-title">Analysis Error</h2>
        <p className="error-text">{error}</p>
        <div className="btn-row">
          <Link to={`/dashboard/${stock}`} className="btn btn-secondary">Back to Dashboard</Link>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>Retry Analysis</button>
        </div>
      </div>
    )
  }

  // Format Heatmap labels and text values
  const cleanFeatures = data.features.map(f => {
    if (f === 'Close') return `${stockName} Close`
    if (f === 'Gold_Close') return 'Gold Spot Price'
    if (f === 'Crude_Close') return 'Brent Crude Oil'
    if (f === 'USD_INR_Close') return 'USD / INR Rate'
    if (f === 'SP500_Close') return 'S&P 500 Index'
    if (f === 'VIX_Close') return 'VIX Fear Index'
    if (f === 'Nifty50_Close') return 'Nifty 50 Index'
    return f
  })

  // Format cell annotation text
  const annotations = []
  for (let i = 0; i < data.features.length; i++) {
    for (let j = 0; j < data.features.length; j++) {
      const val = data.matrix[i][j]
      annotations.push({
        x: cleanFeatures[j],
        y: cleanFeatures[i],
        text: val.toFixed(2),
        font: { color: Math.abs(val) > 0.4 ? 'white' : 'var(--text-primary)', size: 12, weight: 'bold' },
        showarrow: false
      })
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      {/* Header Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
        <div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Macroeconomic Context
          </span>
          <h1 style={{ fontSize: '2rem', fontWeight: 800 }}>
            {stockName} Global Macro Correlations
          </h1>
        </div>
        <Link to={`/dashboard/${stock}`} className="btn btn-secondary">
          ⬅ Back to Dashboard
        </Link>
      </div>

      <div className="card">
        <h2 className="card-title">Pearson Correlation Heatmap</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '20px' }}>
          This heatmap details the strength of linear association (Pearson's r) between the daily Close price of {stockName} and leading global economic proxies. A value of +1.0 indicates perfect positive alignment, -1.0 indicates perfect inverse alignment, and 0.0 indicates no relationship.
        </p>

        <div style={{ width: '100%', overflowX: 'auto', display: 'flex', justifyContent: 'center' }}>
          <Plot
            data={[
              {
                x: cleanFeatures,
                y: cleanFeatures,
                z: data.matrix,
                type: 'heatmap',
                colorscale: [
                  [0, '#ef4444'],   // Dark red (strong negative correlation)
                  [0.5, '#171721'], // Dark neutral
                  [1, '#3b82f6']    // Bright blue (strong positive correlation)
                ],
                showscale: true,
                zmin: -1.0,
                zmax: 1.0
              }
            ]}
            layout={{
              width: 550,
              height: 500,
              margin: { l: 120, r: 20, t: 20, b: 80 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: 'var(--text-secondary)', family: 'var(--font-family)', size: 10 },
              xaxis: {
                tickangle: -45,
                linecolor: 'rgba(255, 255, 255, 0.1)'
              },
              yaxis: {
                autorange: 'reverse', // to match matrix top-to-bottom layout
                linecolor: 'rgba(255, 255, 255, 0.1)'
              },
              annotations: annotations,
              template: 'plotly_dark'
            }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>
    </div>
  )
}

export default MacroHeatmap
