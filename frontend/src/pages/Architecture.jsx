import React from 'react'

function Architecture() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '35px' }}>
      <div>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
          Network Pipeline Flow
        </span>
        <h1 style={{ fontSize: '2rem', fontWeight: 800 }}>Hybrid BiLSTM-MTRAN-TCN Neural Network</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '4px', maxWidth: '850px' }}>
          A breakdown of how the 18 daily features are ingested, processed sequentially, attention-scaled, and filtered via temporal convolutions to generate tomorrow's baseline forecast.
        </p>
      </div>

      <div className="card">
        <h2 className="card-title" style={{ textAlign: 'center', marginBottom: '30px' }}>Data & Network Pipeline Flowchart</h2>
        
        <div className="flowchart">
          
          {/* Step 1: Inputs */}
          <div className="flow-step">
            <div className="flow-step-number">1</div>
            <h3>Multivariate Input Tensors</h3>
            <p>10 consecutive trading days (sequence window) × 18 normalized features (OHLCV + 6 Technical Indicators + 6 Global Macroeconomic Features).</p>
            <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--accent-blue)', marginTop: '8px' }}>
              Shape: [Batch Size, 10, 18]
            </div>
          </div>
          
          <div className="flow-arrow" />

          {/* Step 2: BiLSTM */}
          <div className="flow-step">
            <div className="flow-step-number">2</div>
            <h3>Bidirectional LSTM Block</h3>
            <p>Processes the 10-day sequences forward and backward. Retains short-term sequential patterns and memory cells representing recent price momentum dynamics.</p>
            <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--accent-blue)', marginTop: '8px' }}>
              Extracts bidirectional recurrent hidden states.
            </div>
          </div>

          <div className="flow-arrow" />

          {/* Step 3: MTRAN */}
          <div className="flow-step">
            <div className="flow-step-number">3</div>
            <h3>Modified Transformer Attention (MTRAN)</h3>
            <p>Applies multi-head self-attention on the recurrent states. Learns complex long-range temporal associations (e.g. how a drop on day 2 correlates with a jump on day 9) regardless of sequence distance.</p>
            <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--accent-blue)', marginTop: '8px' }}>
              Outputs attention-weighted contextual embeddings.
            </div>
          </div>

          <div className="flow-arrow" />

          {/* Step 4: TCN */}
          <div className="flow-step">
            <div className="flow-step-number">4</div>
            <h3>Temporal Convolutional Network (TCN)</h3>
            <p>Uses dilated causal 1D convolutional layers to extract local structural features. Causal structure ensures prediction at time T only relies on history [0, T-1] to prevent future data leakage.</p>
            <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--accent-blue)', marginTop: '8px' }}>
              Dilated convolutions expand the receptive field without pooling losses.
            </div>
          </div>

          <div className="flow-arrow" />

          {/* Step 5: Output */}
          <div className="flow-step" style={{ border: '1px solid rgba(16, 185, 129, 0.3)' }}>
            <div className="flow-step-number" style={{ backgroundColor: 'var(--accent-green)', boxShadow: '0 0 10px var(--accent-green)' }}>5</div>
            <h3>Linear Output Head</h3>
            <p>Flattens TCN features and runs them through a linear mapping dense layer to output the single predicted target value: tomorrow's baseline closing price.</p>
            <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--accent-green)', marginTop: '8px' }}>
              Output Shape: [Batch Size, 1] (Predicted Close)
            </div>
          </div>

        </div>
      </div>

      {/* Network Benefits */}
      <h2 className="section-title">Why this hybrid approach works</h2>
      <div className="grid-cols-3">
        <div className="card">
          <h3 className="card-title">1. Recurrent Memory</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.5' }}>
            BiLSTM captures immediate momentum (e.g., strong yesterday gains imply short-term carry-over trends).
          </p>
        </div>
        <div className="card">
          <h3 className="card-title">2. Contextual Attention</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.5' }}>
            Transformer self-attention maps multi-day dependencies, recognizing when indicator signals diverge over time.
          </p>
        </div>
        <div className="card">
          <h3 className="card-title">3. Leak-free Convolutions</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.5' }}>
            TCN's causal convolutions ensure strict temporal ordering, filtering high-frequency noise without look-ahead bias.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Architecture
