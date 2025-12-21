import { useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'

interface BenchmarkResponse {
  gpu_time_ms: number
  gpu_time_fp16_ms: number
  cpu_time_ms: number
  speedup: number
  speedup_fp16: number
}

export default function BenchmarkChart() {
  const [data, setData] = useState<BenchmarkResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const runBenchmark = async () => {
    setLoading(true)
    try {
      const res = await axios.post<BenchmarkResponse>('/api/benchmark')
      setData(res.data)
    } catch (err) {
      console.error("benchmark failed", err)
    } finally {
      setLoading(false)
    }
  }

  const chartData = data ? [
    { name: 'CPU', time: data.cpu_time_ms, color: '#ff4d4f' },
    { name: 'GPU (FP32)', time: data.gpu_time_ms, color: '#52c41a' },
    { name: 'GPU (FP16)', time: data.gpu_time_fp16_ms, color: '#13c2c2' }
  ] : []

  return (
    <div style={{ marginTop: '40px', borderTop: '1px solid #eee', paddingTop: '20px' }}>
      <h2>Performance Benchmark</h2>
      
      <div style={{ marginBottom: '20px', color: '#666', maxWidth: '800px', lineHeight: '1.5' }}>
        <div style={{ backgroundColor: '#fff7e6', padding: '15px', borderRadius: '8px', border: '1px solid #ffd591', marginBottom: '15px' }}>
            <strong>Configuration:</strong>
            <ul style={{ marginTop: '5px', marginBottom: '0', paddingLeft: '20px' }}>
                <li><strong>Model:</strong> LSTM</li>
                <li><strong>Input Size:</strong> 1024</li>
                <li><strong>Hidden Size:</strong> 4096</li>
                <li><strong>Time Steps:</strong> 50</li>
            </ul>
        </div>

        <p>
          This test compares the raw speed of your CPU against the GPU running the exact same LSTM algorithm.
        </p>
        <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
          <li><strong>CPU:</strong> Runs the calculation sequentially. It might take a few seconds.</li>
          <li><strong>GPU (FP32):</strong> Runs in parallel using standard precision. Much faster than CPU.</li>
          <li><strong>GPU (FP16):</strong> Runs using "half precision" on Tensor Cores. This is the fastest mode.</li>
        </ul>
      </div>

      <button 
        onClick={runBenchmark} 
        disabled={loading}
        style={{ 
          padding: '10px 20px', 
          fontSize: '16px', 
          cursor: loading ? 'not-allowed' : 'pointer',
          marginBottom: '20px',
          backgroundColor: '#1890ff',
          color: 'white',
          border: 'none',
          borderRadius: '4px'
        }}
      >
        {loading ? 'Running Benchmark...' : 'Run CPU vs GPU Benchmark'}
      </button>

      {data && (
        <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
          <div style={{ width: '600px', height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Execution Time (ms)', angle: -90, position: 'insideLeft', dy: 50 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="time" name="Execution Time (ms)">
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div>
            <h3 style={{ fontSize: '24px', marginBottom: '10px' }}>Speedup Factor</h3>
            <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#52c41a' }}>
                {data.speedup.toFixed(1)}x
                </div>
                <p style={{ color: '#666', margin: 0 }}>FP32 vs CPU</p>
            </div>
            <div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#13c2c2' }}>
                {data.speedup_fp16.toFixed(1)}x
                </div>
                <p style={{ color: '#666', margin: 0 }}>FP16 vs CPU</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
