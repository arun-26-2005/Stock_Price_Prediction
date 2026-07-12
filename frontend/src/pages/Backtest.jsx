import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import LoadingSpinner from '../components/LoadingSpinner'
import KPICard from '../components/KPICard'

function Backtest() {
  const { stock } = useParams()
  const stockName = stock.toUpperCase()

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)
  const [showExplainers, setShowExplainers] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    
    fetch(`/api/backtest/${stockName}`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to run backtest simulation')
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

  if (loading) return <LoadingSpinner text={`Running walk-forward backtest simulation for ${stockName}...`} />

  if (error) {
    return (
      <div className="error-container">
        <h2 className="error-title">Simulation Error</h2>
        <p className="error-text">{error}</p>
        <div className="btn-row">
          <Link to={`/dashboard/${stock}`} className="btn btn-secondary">Back to Dashboard</Link>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>Retry Simulation</button>
        </div>
      </div>
    )
  }

  // Prep Plotly lines
  const traceStrategy = {
    x: data.dates,
    y: data.portfolio_values,
    type: 'scatter',
    mode: 'lines',
    name: `Model Strategy (Return: ${data.total_return.toFixed(2)}%)`,
    line: { color: '#3b82f6', width: 2.5 }
  }

  const traceBenchmark = {
    x: data.dates,
    y: data.benchmark_values,
    type: 'scatter',
    mode: 'lines',
    name: `Buy & Hold Benchmark (Return: ${data.bench_return.toFixed(2)}%)`,
    line: { color: '#f59e0b', width: 1.5, dash: 'dash' }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      {/* Header Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
        <div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Historical Performance
          </span>
          <h1 style={{ fontSize: '2rem', fontWeight: 800 }}>
            {stockName} Strategy Backtester
          </h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <label className="control-label" style={{ background: 'rgba(255,255,255,0.03)', padding: '8px 16px', borderRadius: '20px', border: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', userSelect: 'none' }}>
            <input 
              type="checkbox" 
              className="control-input" 
              checked={showExplainers} 
              onChange={(e) => setShowExplainers(e.target.checked)} 
              style={{ width: '15px', height: '15px' }}
            />
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>Explain Metrics (❔)</span>
          </label>
          <Link to={`/dashboard/${stock}`} className="btn btn-secondary">
            ⬅ Back to Prediction
          </Link>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="kpi-grid">
        <KPICard 
          title="Strategy Return" 
          value={`${data.total_return.toFixed(2)}%`}
          subtitle="Model predictive trades"
          trend={data.total_return >= data.bench_return ? 'up' : 'down'}
          showExplainer={showExplainers}
          explainerText="How much profit/loss you made by buying and selling based on AI predictions."
          sparklineData={data && data.portfolio_values ? data.portfolio_values.slice(-15) : null}
          sparklineColor={data.total_return >= data.bench_return ? 'var(--accent-green)' : 'var(--accent-red)'}
        />
        <KPICard 
          title="Benchmark Return" 
          value={`${data.bench_return.toFixed(2)}%`}
          subtitle="Pure Buy & Hold"
          trend="neutral"
          showExplainer={showExplainers}
          explainerText="How much profit/loss you made by simply buying the stock and holding it the entire time."
          sparklineData={data && data.benchmark_values ? data.benchmark_values.slice(-15) : null}
          sparklineColor="var(--accent-yellow)"
        />
        <KPICard 
          title="Sharpe Ratio" 
          value={data.sharpe_ratio.toFixed(2)} 
          subtitle="Risk-adjusted return"
          trend={data.sharpe_ratio >= 0 ? 'up' : 'down'}
          showExplainer={showExplainers}
          explainerText="Risk score. Above 0.5 means you are getting stable returns without taking wild, stressful risks."
        />
        <KPICard 
          title="Max Drawdown" 
          value={`${data.max_drawdown.toFixed(2)}%`} 
          subtitle="Max peak-to-trough loss"
          trend={data.max_drawdown <= data.bench_max_drawdown ? 'up' : 'down'}
          showExplainer={showExplainers}
          explainerText="Worst-case drop. The maximum drop from a peak you would have suffered historically."
        />
        <KPICard 
          title="Win Rate" 
          value={`${data.win_rate.toFixed(1)}%`} 
          subtitle={`${data.num_trades} executed trades`}
          trend={data.win_rate >= 50 ? 'up' : 'neutral'}
          showExplainer={showExplainers}
          explainerText="The percentage of executed trades that closed in profit. High win rate means high confidence."
        />
      </div>

      {/* Equity Curve Chart */}
      <div className="card">
        <h2 className="card-title">Equity Curve: Strategy vs Benchmark Simulation</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '20px' }}>
          Portfolio equity curves starting with an initial capital of INR 100,000. Simulation triggers a buy/hold when predicted tomorrow's close price is higher than today's actual close by a threshold of 0.25%, incorporating a 0.1% round-trip trading transaction fee.
        </p>

        <div style={{ width: '100%', overflowX: 'auto' }}>
          <Plot
            data={[traceStrategy, traceBenchmark]}
            layout={{
              autosize: true,
              height: 450,
              margin: { l: 70, r: 20, t: 10, b: 40 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: 'var(--text-secondary)', family: 'var(--font-family)' },
              xaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                title: 'Date'
              },
              yaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                title: 'Portfolio Value (INR)'
              },
              legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1
              },
              template: 'plotly_dark'
            }}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      </div>
    </div>
  )
}

export default Backtest
