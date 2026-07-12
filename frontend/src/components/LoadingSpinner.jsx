import React from 'react'

function LoadingSpinner({ text = 'Analyzing real-time indicators...' }) {
  return (
    <div className="spinner-container">
      <div className="spinner-orbital">
        <div className="spinner-ring spinner-ring-1" />
        <div className="spinner-ring spinner-ring-2" />
        <div className="spinner-core" />
      </div>
      <span className="spinner-text">
        {text}<span className="spinner-dots" />
      </span>
    </div>
  )
}

export default LoadingSpinner
