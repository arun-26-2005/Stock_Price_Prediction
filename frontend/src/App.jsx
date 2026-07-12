import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import Backtest from './pages/Backtest'
import MacroHeatmap from './pages/MacroHeatmap'
import ModelComparison from './pages/ModelComparison'
import Architecture from './pages/Architecture'
import Login from './pages/Login'
import AdminDashboard from './pages/AdminDashboard'

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <div className="page-container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard/:stock" element={<Dashboard />} />
          <Route path="/backtest/:stock" element={<Backtest />} />
          <Route path="/macro/:stock" element={<MacroHeatmap />} />
          <Route path="/models" element={<ModelComparison />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/login" element={<Login />} />
          <Route path="/admin" element={<AdminDashboard />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

export default App
