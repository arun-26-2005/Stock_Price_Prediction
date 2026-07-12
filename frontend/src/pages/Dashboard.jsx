import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import LoadingSpinner from '../components/LoadingSpinner'
import SentimentGauge from '../components/SentimentGauge'
import KPICard from '../components/KPICard'
import NewsCard from '../components/NewsCard'

function Dashboard() {
  const { stock } = useParams()
  const stockName = stock.toUpperCase()

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  
  const [predData, setPredData] = useState(null)
  const [histData, setHistData] = useState(null)
  
  // Chart toggles
  const [showMA5, setShowMA5] = useState(true)
  const [showMA10, setShowMA10] = useState(false)
  const [showMA20, setShowMA20] = useState(false)
  const [showBB, setShowBB] = useState(false)

  // Advisor & explainer states
  const [ownsStock, setOwnsStock] = useState(false)
  const [buyPrice, setBuyPrice] = useState('')
  const [quantity, setQuantity] = useState('')
  const [advisorResult, setAdvisorResult] = useState(null)
  const [showExplainers, setShowExplainers] = useState(false)

  // Retraining state
  const [retrainStatus, setRetrainStatus] = useState("idle")
  const [retrainMessage, setRetrainMessage] = useState("")

  useEffect(() => {
    setLoading(true)
    setError(null)
    
    // Fetch prediction and history in parallel
    Promise.all([
      fetch(`/api/predict/${stockName}`).then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch prediction for ${stockName}`)
        return res.json()
      }),
      fetch(`/api/history/${stockName}`).then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch history for ${stockName}`)
        return res.json()
      })
    ])
      .then(([pred, hist]) => {
        if (pred.error) {
          throw new Error(pred.error)
        }
        setPredData(pred)
        setHistData(hist)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [stockName])

  if (loading) return <LoadingSpinner text={`Computing live prediction for ${stockName}...`} />

  if (error) {
    return (
      <div className="error-container">
        <h2 className="error-title">Inference Error</h2>
        <p className="error-text">{error}</p>
        <div className="btn-row">
          <Link to="/" className="btn btn-secondary">Back to Home</Link>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>Retry</button>
        </div>
      </div>
    )
  }

  // Prep Plotly Price Series
  const chartData = []

  // 1. Candlestick Chart (Close Price Line or Candlestick)
  // To keep it clean and premium, we'll draw a main trace for Close price
  chartData.push({
    x: histData.dates,
    y: histData.close,
    type: 'scatter',
    mode: 'lines',
    name: 'Close Price',
    line: { color: '#3b82f6', width: 2 },
  })

  // Overlay MA5
  if (showMA5) {
    chartData.push({
      x: histData.dates,
      y: histData.ma5,
      type: 'scatter',
      mode: 'lines',
      name: 'MA 5',
      line: { color: '#10b981', width: 1.5, dash: 'solid' },
    })
  }

  // Overlay MA10
  if (showMA10) {
    chartData.push({
      x: histData.dates,
      y: histData.ma10,
      type: 'scatter',
      mode: 'lines',
      name: 'MA 10',
      line: { color: '#f59e0b', width: 1.5, dash: 'solid' },
    })
  }

  // Overlay MA20
  if (showMA20) {
    chartData.push({
      x: histData.dates,
      y: histData.ma20,
      type: 'scatter',
      mode: 'lines',
      name: 'MA 20',
      line: { color: '#8b5cf6', width: 1.5, dash: 'solid' },
    })
  }

  // Overlay Bollinger Bands
  if (showBB) {
    chartData.push({
      x: histData.dates,
      y: histData.bb_high,
      type: 'scatter',
      mode: 'lines',
      name: 'BB Upper',
      line: { color: 'rgba(255, 255, 255, 0.25)', width: 1, dash: 'dash' },
    })
    chartData.push({
      x: histData.dates,
      y: histData.bb_low,
      type: 'scatter',
      mode: 'lines',
      name: 'BB Lower',
      line: { color: 'rgba(255, 255, 255, 0.25)', width: 1, dash: 'dash' },
      fill: 'tonexty',
      fillcolor: 'rgba(255, 255, 255, 0.02)',
    })
  }

  const handleCalculateAdvice = (e) => {
    e.preventDefault()
    
    const currentPrice = predData.last_close
    const targetPrice = predData.adjusted_price
    const predictedChange = predData.change_pct
    
    let recommendation = ''
    let reasoning = ''
    let badgeClass = ''
    let details = {}

    if (!ownsStock) {
      if (predictedChange > 0.5) {
        recommendation = 'BUY'
        badgeClass = 'badge-success'
        reasoning = `The model predicts the stock price will rise tomorrow to INR ${targetPrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (+${predictedChange.toFixed(2)}%). This is a strong positive movement signal, indicating a good entry point to buy.`
      } else if (predictedChange < -0.5) {
        recommendation = 'WAIT / AVOID'
        badgeClass = 'badge-danger'
        reasoning = `The model predicts the price will drop tomorrow to INR ${targetPrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (${predictedChange.toFixed(2)}%). We recommend waiting for a cheaper price to enter.`
      } else {
        recommendation = 'MONITOR'
        badgeClass = 'badge-warning'
        reasoning = `The model predicts sideways movement tomorrow (INR ${targetPrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}). There is no strong upward or downward signal. Keep this stock on your watch list for now.`
      }
    } else {
      const priceNum = parseFloat(buyPrice)
      const qtyNum = parseFloat(quantity) || 1
      
      if (isNaN(priceNum) || priceNum <= 0) {
        alert('Please enter a valid buy price.')
        return
      }

      const currentReturn = ((currentPrice - priceNum) / priceNum) * 100
      const currentProfit = (currentPrice - priceNum) * qtyNum
      const isCurrentlyInProfit = currentPrice >= priceNum

      details = {
        currentReturn,
        currentProfit,
        isCurrentlyInProfit
      }

      if (predictedChange > 0.25) {
        if (isCurrentlyInProfit) {
          recommendation = 'HOLD'
          badgeClass = 'badge-success'
          reasoning = `You are currently in profit (+${currentReturn.toFixed(2)}%), and the model predicts tomorrow's price will rise further. We recommend holding your position to maximize your returns.`
        } else {
          recommendation = 'HOLD / WAIT OUT'
          badgeClass = 'badge-warning'
          reasoning = `You are currently down (-${Math.abs(currentReturn).toFixed(2)}%), but the model predicts a recovery trend tomorrow. We recommend holding to see if you can break even or exit at a better price.`
        }
      } else if (predictedChange < -0.25) {
        if (isCurrentlyInProfit) {
          recommendation = 'SELL / TAKE PROFIT'
          badgeClass = 'badge-danger'
          reasoning = `You are currently in profit (+${currentReturn.toFixed(2)}%), but the model predicts a drop tomorrow. We recommend selling now to lock in your profits before the drop.`
        } else {
          recommendation = 'SELL / CUT LOSS'
          badgeClass = 'badge-danger'
          reasoning = `You are currently down (-${Math.abs(currentReturn).toFixed(2)}%), and the model predicts the price will drop further tomorrow. We recommend selling now to protect your remaining capital and prevent a larger loss.`
        }
      } else {
        recommendation = 'HOLD'
        badgeClass = 'badge-neutral'
        reasoning = `The model predicts tomorrow's price will remain stable. Since there are no strong upward or downward triggers, we recommend holding your position.`
      }
    }

    setAdvisorResult({
      recommendation,
      badgeClass,
      reasoning,
      details,
      ownsStock
    })
  }

  const handleRetrain = () => {
    setRetrainStatus("running")
    setRetrainMessage("Initializing sync with Yahoo Finance...")

    fetch(`/api/retrain/${stockName}`, { method: "POST" })
      .then((res) => res.json())
      .then((resData) => {
        if (resData.error) throw new Error(resData.error)
        
        // Start polling status
        const intervalId = setInterval(() => {
          fetch(`/api/retrain/status/${stockName}`)
            .then((res) => res.json())
            .then((statusData) => {
              if (statusData.status === "success") {
                clearInterval(intervalId)
                setRetrainStatus("success")
                setRetrainMessage(statusData.message)
                setTimeout(() => {
                  window.location.reload()
                }, 2000)
              } else if (statusData.status === "failed") {
                clearInterval(intervalId)
                setRetrainStatus("failed")
                setRetrainMessage(statusData.message)
              } else {
                setRetrainMessage(statusData.message)
              }
            })
            .catch((err) => {
              clearInterval(intervalId)
              setRetrainStatus("failed")
              setRetrainMessage("Error checking retraining progress.")
            })
        }, 2000)
      })
      .catch((err) => {
        setRetrainStatus("failed")
        setRetrainMessage(err.message || "Failed to trigger retraining.")
      })
  }

  // Determine trend color direction
  const changePct = predData.change_pct
  const isUp = changePct >= 0
  const changeColor = isUp ? 'var(--accent-green)' : 'var(--accent-red)'

  return (
    <div className="dashboard-layout">
      {/* Top Header Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
        <div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Live Forecast Workspace
          </span>
          <h1 style={{ fontSize: '2rem', fontWeight: 800 }}>
            {predData.stock} prediction: {predData.symbol}
          </h1>
        </div>
        <div className="btn-row" style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
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
          <Link to={`/backtest/${stock}`} className="btn btn-secondary">
            📈 Backtest Strategy
          </Link>
          <Link to={`/macro/${stock}`} className="btn btn-secondary">
            🌍 Macro Analysis
          </Link>
          <button 
            onClick={handleRetrain} 
            disabled={retrainStatus === "running"} 
            className="btn btn-primary"
            style={{ minWidth: '130px', justifyContent: 'center' }}
          >
            {retrainStatus === "running" ? "⚡ Retraining..." : "🔄 Sync & Retrain"}
          </button>
        </div>
      </div>

      {/* Retraining status notification banner */}
      {retrainStatus !== "idle" && (
        <div style={{
          background: retrainStatus === "running" ? 'rgba(79,140,255,0.08)' : (retrainStatus === "success" ? 'rgba(0,212,170,0.08)' : 'rgba(255,71,87,0.08)'),
          border: `1px solid ${retrainStatus === "running" ? 'rgba(79,140,255,0.2)' : (retrainStatus === "success" ? 'rgba(0,212,170,0.2)' : 'rgba(255,71,87,0.2)')}`,
          color: retrainStatus === "running" ? 'var(--accent-blue)' : (retrainStatus === "success" ? 'var(--accent-green)' : 'var(--accent-red)'),
          padding: '12px 20px',
          borderRadius: '10px',
          fontSize: '0.88rem',
          fontWeight: 600,
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }}>
          {retrainStatus === "running" && <span className="pulse-dot" style={{ backgroundColor: 'var(--accent-blue)', boxShadow: '0 0 8px var(--accent-blue)' }} />}
          {retrainStatus === "success" && <span>✅</span>}
          {retrainStatus === "failed" && <span>⚠️</span>}
          <span>{retrainMessage}</span>
        </div>
      )}

      {/* Top Dashboard Panels */}
      <div className="top-panel">
        {/* Prediction Main Block */}
        <div className="card predictions-detail-card">
          <div>
            <h2 className="card-title">Forecast Overview (Next Session)</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Base prediction generated by BiLSTM-MTRAN-TCN network, dynamically adjusted by today's consolidated sentiment score.
            </p>
            {showExplainers && (
              <div style={{ margin: '12px 0 0 0', padding: '12px', background: 'rgba(59,130,246,0.08)', border: '1px solid var(--accent-blue-glow)', borderRadius: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                💡 <strong>Tomorrow's Forecast Target:</strong> The AI predicts tomorrow's stock price based on chart patterns, global macro indexes, and news.
                <div style={{ marginTop: '6px' }}>• <strong>Baseline:</strong> The raw technical chart forecasting price.</div>
                <div>• <strong>News-Adjusted:</strong> The final target price after shifting it based on today's news sentiment shock.</div>
              </div>
            )}
          </div>
          
          <div className="prediction-hero-row">
            <div className="prediction-col">
              <span className="prediction-label">Baseline forecast</span>
              <span className="prediction-val">INR {predData.baseline_price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            </div>
            <div className="prediction-arrow">➔</div>
            <div className="prediction-col">
              <span className="prediction-label">News-Adjusted Target</span>
              <span className="prediction-val highlighted">INR {predData.adjusted_price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginTop: '20px' }}>
            <div style={{ flex: 1, minWidth: '150px' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block', marginBottom: '4px' }}>
                Forecast Direction
              </span>
              <span style={{ fontSize: '1.4rem', fontWeight: 800, color: changeColor }}>
                {isUp ? '▲ BULLISH' : '▼ BEARISH'} ({changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%)
              </span>
            </div>
            <div style={{ flex: 1, minWidth: '150px' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block', marginBottom: '4px' }}>
                Last Close Date
              </span>
              <span style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                {predData.last_date}
              </span>
            </div>
          </div>
        </div>

        {/* Sentiment Gauge Block */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <h2 className="card-title" style={{ textAlign: 'center', marginBottom: '10px' }}>Consolidated Sentiment</h2>
          {showExplainers && (
            <div style={{ margin: '0 0 12px 0', padding: '10px', background: 'rgba(59,130,246,0.08)', border: '1px solid var(--accent-blue-glow)', borderRadius: '8px', fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.4', textAlign: 'center' }}>
              💡 <strong>News Sentiment:</strong> How positive or negative today's headlines are. We read them via NLP to shift the price prediction up or down.
            </div>
          )}
          <SentimentGauge score={predData.sentiment_score} />
        </div>
      </div>

      {/* Reusable Advisor Card inserted right below Top Panels */}
      {/* Personalized Advisor Section */}
      {/* Personalized Advisor Section */}
      <div className="card" style={{ padding: '30px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', borderBottom: '1px solid var(--border-color)', paddingBottom: '16px' }}>
          <div>
            <h2 className="card-title" style={{ margin: 0, fontSize: '1.25rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>🤖 Personalized Stock Action Advisor</span>
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginTop: '4px' }}>
              Query tomorrow's optimized position strategy based on custom portfolio holds.
            </p>
          </div>
          <span className="stock-card-badge badge-neutral" style={{ fontSize: '0.72rem', letterSpacing: '0.05em' }}>AI Engine v2.0</span>
        </div>

        <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap' }}>
          {/* Left Column: Form Controls */}
          <form onSubmit={handleCalculateAdvice} style={{ flex: 1, minWidth: '280px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Your Portfolio Status
              </span>
              <div className="segmented-control" style={{ width: 'fit-content' }}>
                <button
                  type="button"
                  className={`segmented-button ${ownsStock ? 'active' : ''}`}
                  onClick={() => { setOwnsStock(true); setAdvisorResult(null); }}
                >
                  💼 I Own Shares
                </button>
                <button
                  type="button"
                  className={`segmented-button ${!ownsStock ? 'active' : ''}`}
                  onClick={() => { setOwnsStock(false); setAdvisorResult(null); }}
                >
                  👁️ Watcher Only
                </button>
              </div>
            </div>

            {ownsStock ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Average Buy Price (INR)</label>
                  <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                    <span style={{ position: 'absolute', left: '12px', color: 'var(--text-muted)', fontSize: '0.95rem', fontWeight: 600 }}>₹</span>
                    <input
                      type="number"
                      step="any"
                      placeholder="e.g. 3150.50"
                      value={buyPrice}
                      onChange={(e) => { setBuyPrice(e.target.value); setAdvisorResult(null); }}
                      required
                      style={{
                        width: '100%',
                        background: 'rgba(255,255,255,0.02)',
                        border: '1px solid var(--glass-border)',
                        borderRadius: '10px',
                        padding: '12px 12px 12px 28px',
                        color: 'white',
                        outline: 'none',
                        fontSize: '0.95rem'
                      }}
                    />
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Quantity Held</label>
                  <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                    <span style={{ position: 'absolute', left: '12px', color: 'var(--text-muted)', fontSize: '0.95rem' }}>📦</span>
                    <input
                      type="number"
                      placeholder="e.g. 25"
                      value={quantity}
                      onChange={(e) => { setQuantity(e.target.value); setAdvisorResult(null); }}
                      required
                      style={{
                        width: '100%',
                        background: 'rgba(255,255,255,0.02)',
                        border: '1px solid var(--glass-border)',
                        borderRadius: '10px',
                        padding: '12px 12px 12px 34px',
                        color: 'white',
                        outline: 'none',
                        fontSize: '0.95rem'
                      }}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px dashed var(--glass-border)', borderRadius: '10px', padding: '20px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                💡 You are simulating a new position. The model will advise you based on optimal Entry price thresholds.
              </div>
            )}

            <button type="submit" className="btn btn-primary" style={{ padding: '12px 24px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', width: '100%', fontSize: '0.9rem' }}>
              ⚡ Run AI Portfolio Advice
            </button>
          </form>

          {/* Right Column: Terminal Screen */}
          <div style={{
            flex: 1.2,
            minWidth: '280px',
            background: 'rgba(5, 6, 10, 0.4)',
            border: '1px solid var(--glass-border)',
            borderRadius: '12px',
            padding: '24px',
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            boxShadow: 'inset 0 0 20px rgba(0,0,0,0.4)',
            minHeight: '260px'
          }}>
            {!advisorResult ? (
              <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: 'var(--text-muted)' }}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v1.244c0 .462-.375.837-.837.837H7.5m3 12h.008v.008H10.5m3-3h.008v.008H13.5m3-6H16.5m-6 0H9.75M9 10.5h.008v.008H9V10.5Zm6 1.5h.008v.008H15V12Zm-3-1.5h.008v.008H12V10.5Zm3 3.5h.008v.008H15V14Zm-3 0h.008v.008H12V14Zm-3-3h.008v.008H9V11Zm3-3h.008v.008H12V8Zm3 0h.008v.008H15V8Zm-6 6h.008v.008H9V14Zm3 3h.008v.008H12V17Zm3 0h.008v.008H15V17Z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 21V5.25A2.25 2.25 0 0 0 17.25 3H6.75A2.25 2.25 0 0 0 4.5 5.25V21m15 0h-15" />
                </svg>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                  Awaiting Input
                </span>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', maxWidth: '280px', lineHeight: '1.4' }}>
                  Input your position metrics on the left and trigger the analyzer to read out the recommendation.
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '15px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>
                      Calculated Action Recommendation
                    </span>
                    <span
                      className={`stock-card-badge ${advisorResult.badgeClass}`}
                      style={{
                        fontSize: '1.1rem',
                        padding: '6px 16px',
                        borderRadius: '30px',
                        display: 'inline-block',
                        fontWeight: 800,
                        boxShadow: `0 0 15px ${advisorResult.badgeClass === 'badge-success' ? 'var(--accent-green-glow)' : advisorResult.badgeClass === 'badge-danger' ? 'var(--accent-red-glow)' : 'var(--accent-yellow-glow)'}`
                      }}
                    >
                      {advisorResult.recommendation}
                    </span>
                  </div>

                  {advisorResult.ownsStock && (
                    <div style={{ display: 'flex', gap: '20px', background: 'rgba(255,255,255,0.02)', padding: '8px 14px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                      <div>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontWeight: 600 }}>Unrealized ROR</span>
                        <span style={{ fontSize: '1rem', fontWeight: 800, color: advisorResult.details.isCurrentlyInProfit ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                          {advisorResult.details.isCurrentlyInProfit ? '▲' : '▼'} {Math.abs(advisorResult.details.currentReturn).toFixed(2)}%
                        </span>
                      </div>
                      <div style={{ width: '1px', background: 'var(--glass-border)' }} />
                      <div>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontWeight: 600 }}>Net Position PnL</span>
                        <span style={{ fontSize: '1rem', fontWeight: 800, color: advisorResult.details.isCurrentlyInProfit ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                          {advisorResult.details.isCurrentlyInProfit ? '+' : '-'} ₹{Math.abs(advisorResult.details.currentProfit).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                <div style={{
                  flex: 1,
                  background: 'rgba(0,0,0,0.2)',
                  border: '1px solid rgba(255,255,255,0.03)',
                  borderRadius: '8px',
                  padding: '16px',
                  fontSize: '0.88rem',
                  lineHeight: '1.5',
                  color: 'var(--text-primary)',
                  position: 'relative'
                }}>
                  <div style={{ borderLeft: '3px solid var(--accent-blue)', paddingLeft: '12px', height: '100%' }}>
                    <strong style={{ color: 'var(--accent-blue)', display: 'block', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px', fontWeight: 700 }}>
                      💡 Advisor Explanation
                    </strong>
                    {advisorResult.reasoning}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="kpi-grid">
        <KPICard 
          title="Last Close Price" 
          value={`INR ${predData.last_close.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`} 
          subtitle="Real-time market price"
          showExplainer={showExplainers}
          explainerText="The price at which the stock finished trading in the last session."
          sparklineData={histData && histData.close ? histData.close.slice(-15) : null}
          sparklineColor={histData && histData.close && histData.close[histData.close.length - 1] >= histData.close[histData.close.length - 15] ? 'var(--accent-green)' : 'var(--accent-red)'}
        />
        <KPICard 
          title="Predicted Change" 
          value={`${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`}
          subtitle="Sentiment-Adjusted delta"
          trend={isUp ? 'up' : 'down'}
          showExplainer={showExplainers}
          explainerText="The percentage gain or loss predicted by the AI model for tomorrow."
          sparklineData={histData && histData.close ? [...histData.close.slice(-14), predData.adjusted_price] : null}
          sparklineColor={isUp ? 'var(--accent-green)' : 'var(--accent-red)'}
        />
        <KPICard 
          title="News Articles" 
          value={predData.articles.length} 
          subtitle="Analyzed last 24h"
          trend={predData.articles.length > 0 ? 'up' : 'neutral'}
          showExplainer={showExplainers}
          explainerText="Total news stories collected today and fed to the sentiment model."
        />
        <KPICard 
          title="Sentiment Score" 
          value={predData.sentiment_score.toFixed(2)} 
          subtitle={predData.sentiment_label}
          trend={predData.sentiment_score >= 0.15 ? 'up' : (predData.sentiment_score <= -0.15 ? 'down' : 'neutral')}
          showExplainer={showExplainers}
          explainerText="Range -1 (very negative news) to +1 (very positive). Feeds directly into tomorrow's prediction."
        />
      </div>

      {/* Historical Price Chart with overlays */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '15px' }}>
          <h2 className="card-title" style={{ marginBottom: 0 }}>Historical Chart & Overlay Indicators</h2>
          <div className="chart-controls">
            <label className="control-label">
              <input 
                type="checkbox" 
                className="control-input" 
                checked={showMA5} 
                onChange={(e) => setShowMA5(e.target.checked)} 
              />
              MA5 (Green)
            </label>
            <label className="control-label">
              <input 
                type="checkbox" 
                className="control-input" 
                checked={showMA10} 
                onChange={(e) => setShowMA10(e.target.checked)} 
              />
              MA10 (Orange)
            </label>
            <label className="control-label">
              <input 
                type="checkbox" 
                className="control-input" 
                checked={showMA20} 
                onChange={(e) => setShowMA20(e.target.checked)} 
              />
              MA20 (Purple)
            </label>
            <label className="control-label">
              <input 
                type="checkbox" 
                className="control-input" 
                checked={showBB} 
                onChange={(e) => setShowBB(e.target.checked)} 
              />
              Bollinger Bands
            </label>
          </div>
        </div>

        {/* Plotly Container */}
        <div style={{ width: '100%', overflowX: 'auto' }}>
          <Plot
            data={chartData}
            layout={{
              autosize: true,
              height: 450,
              margin: { l: 60, r: 20, t: 10, b: 40 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: 'var(--text-secondary)', family: 'var(--font-family)' },
              xaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                title: 'Trading Date'
              },
              yaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                title: 'Price (INR)'
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

      {/* Subplots Row (RSI / MACD) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '30px' }}>
        {/* RSI Chart */}
        <div className="card">
          <h2 className="card-title">RSI (14) Momentum Indicator</h2>
          <Plot
            data={[
              {
                x: histData.dates,
                y: histData.rsi,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI',
                line: { color: '#ffa502', width: 1.5 }
              },
              // Overbought guide line (70)
              {
                x: [histData.dates[0], histData.dates[histData.dates.length - 1]],
                y: [70, 70],
                type: 'scatter',
                mode: 'lines',
                name: 'Overbought (70)',
                line: { color: 'rgba(239, 68, 68, 0.4)', width: 1.2, dash: 'dot' }
              },
              // Oversold guide line (30)
              {
                x: [histData.dates[0], histData.dates[histData.dates.length - 1]],
                y: [30, 30],
                type: 'scatter',
                mode: 'lines',
                name: 'Oversold (30)',
                line: { color: 'rgba(16, 185, 129, 0.4)', width: 1.2, dash: 'dot' }
              }
            ]}
            layout={{
              autosize: true,
              height: 200,
              margin: { l: 60, r: 20, t: 10, b: 40 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: 'var(--text-secondary)', family: 'var(--font-family)' },
              xaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)'
              },
              yaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                range: [10, 90]
              },
              showlegend: false,
              template: 'plotly_dark'
            }}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      </div>

      {/* News Sentiment Section */}
      <div>
        <h2 className="section-title">Live News & NLP Scrapes (Last 24 Hours)</h2>
        {predData.articles.length === 0 ? (
          <div className="card" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No recent articles matching company symbol parsed by yfinance. Defaulting to neutral sentiment override.
          </div>
        ) : (
          <div className="news-grid">
            {predData.articles.map((art, idx) => (
              <NewsCard
                key={idx}
                title={art.title}
                publisher={art.publisher}
                sentiment={art.sentiment}
                confidence={art.confidence}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
