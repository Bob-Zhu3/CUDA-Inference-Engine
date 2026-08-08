import { useState } from 'react'
import axios from 'axios'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

interface BenchmarkResponse {
  gpu_time_ms: number
  gpu_time_fp16_ms: number
  gpu_end_to_end_ms: number
  gpu_end_to_end_fp16_ms: number
  cpu_time_ms: number
  speedup: number
  speedup_fp16: number
  fp32_max_abs_error: number
  fp16_max_abs_error: number
  verified: boolean
  iterations: number
  device: string
  configuration: {
    input_size: number
    hidden_size: number
    sequence_length: number
    parameter_count: number
  }
}

export default function BenchmarkChart() {
  const [data, setData] = useState<BenchmarkResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const runBenchmark = async () => {
    setLoading(true)
    setError('')
    try {
      const response = await axios.post<BenchmarkResponse>('/api/benchmark')
      setData(response.data)
    } catch (requestError) {
      const message = axios.isAxiosError(requestError)
        ? String(requestError.response?.data?.detail ?? requestError.message)
        : 'Unknown benchmark error'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const chartData = data
    ? [
        { name: 'CPU reference', time: data.cpu_time_ms, color: '#ef4444' },
        { name: 'CUDA FP32', time: data.gpu_time_ms, color: '#22c55e' },
        { name: 'CUDA FP16', time: data.gpu_time_fp16_ms, color: '#06b6d4' },
      ]
    : []

  return (
    <section
      style={{
        marginTop: '42px',
        paddingTop: '28px',
        borderTop: '2px solid #e5e7eb',
      }}
    >
      <h2>Validated CPU vs GPU benchmark</h2>
      <p style={{ maxWidth: '900px', color: '#4b5563', lineHeight: 1.6 }}>
        The same single-layer LSTM runs on a straightforward single-threaded
        CPU reference, CUDA FP32, and mixed FP16. Weights are loaded once.
        Before reporting speed, the benchmark confirms that both GPU outputs
        are numerically close to the CPU result. GPU bars show the median of
        repeated compute-only measurements.
      </p>

      <button
        onClick={runBenchmark}
        disabled={loading}
        style={{
          padding: '11px 20px',
          border: 0,
          borderRadius: '6px',
          background: '#0f766e',
          color: 'white',
          fontSize: '16px',
          cursor: loading ? 'wait' : 'pointer',
          opacity: loading ? 0.7 : 1,
        }}
      >
        {loading ? 'Validating and benchmarking…' : 'Run validated benchmark'}
      </button>

      {loading && (
        <p style={{ color: '#6b7280' }}>
          The first run initializes the larger model, then executes the CPU
          reference and repeated GPU measurements.
        </p>
      )}

      {error && (
        <p
          role="alert"
          style={{
            padding: '12px',
            borderRadius: '6px',
            background: '#fee2e2',
            color: '#991b1b',
          }}
        >
          {error}
        </p>
      )}

      {data && (
        <>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '12px',
              margin: '22px 0',
            }}
          >
            <Stat label="GPU" value={data.device} />
            <Stat
              label="Configuration"
              value={`${data.configuration.input_size} → ${data.configuration.hidden_size} × ${data.configuration.sequence_length} steps`}
            />
            <Stat
              label="LSTM parameters"
              value={data.configuration.parameter_count.toLocaleString()}
            />
            <Stat
              label="Correctness"
              value={data.verified ? 'PASS' : 'FAIL'}
              color={data.verified ? '#15803d' : '#b91c1c'}
            />
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(500px, 2fr) minmax(240px, 1fr)',
              gap: '28px',
              alignItems: 'center',
            }}
          >
            <div style={{ height: '330px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis
                    label={{
                      value: 'Compute time (ms)',
                      angle: -90,
                      position: 'insideLeft',
                    }}
                  />
                  <Tooltip formatter={(value: number) => `${value.toFixed(3)} ms`} />
                  <Bar dataKey="time">
                    {chartData.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div>
              <Speedup
                label="FP32 vs CPU"
                value={data.speedup}
                color="#16a34a"
              />
              <Speedup
                label="FP16 vs CPU"
                value={data.speedup_fp16}
                color="#0891b2"
              />
            </div>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '12px',
              marginTop: '20px',
            }}
          >
            <Stat
              label="FP32 max difference"
              value={data.fp32_max_abs_error.toExponential(3)}
            />
            <Stat
              label="FP16 max difference"
              value={data.fp16_max_abs_error.toExponential(3)}
            />
            <Stat
              label="FP32 end-to-end"
              value={`${data.gpu_end_to_end_ms.toFixed(3)} ms`}
            />
            <Stat
              label="FP16 end-to-end"
              value={`${data.gpu_end_to_end_fp16_ms.toFixed(3)} ms`}
            />
          </div>

          <p style={{ color: '#6b7280', fontSize: '14px' }}>
            Compute time excludes Python conversion and host/GPU transfer.
            End-to-end time includes the per-request input/output path but
            still excludes the one-time model-weight load. GPU medians use{' '}
            {data.iterations} measured runs.
          </p>
        </>
      )}
    </section>
  )
}

function Stat({
  label,
  value,
  color = '#172033',
}: {
  label: string
  value: string
  color?: string
}) {
  return (
    <div
      style={{
        padding: '14px',
        border: '1px solid #d1d5db',
        borderRadius: '7px',
        background: 'white',
      }}
    >
      <div style={{ color: '#6b7280', fontSize: '13px' }}>{label}</div>
      <strong style={{ color, display: 'block', marginTop: '5px' }}>
        {value}
      </strong>
    </div>
  )
}

function Speedup({
  label,
  value,
  color,
}: {
  label: string
  value: number
  color: string
}) {
  return (
    <div style={{ marginBottom: '22px' }}>
      <div style={{ fontSize: '36px', fontWeight: 700, color }}>
        {value.toFixed(1)}×
      </div>
      <div style={{ color: '#6b7280' }}>{label}</div>
    </div>
  )
}
