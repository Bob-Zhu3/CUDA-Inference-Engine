import { useState } from 'react'
import axios from 'axios'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import BenchmarkChart from './BenchmarkChart'

interface LstmResponse {
  output: number[]
  execution_time_ms: number
}

interface AttentionResponse {
  output: number[]
  attention_weights: number[]
  execution_time_ms: number
}

const INPUT_SIZE = 4
const SEQUENCE_LENGTH = 5

function App() {
  const [inputSequence, setInputSequence] = useState<number[][]>([])
  const [lstmOutput, setLstmOutput] = useState<number[]>([])
  const [attentionOutput, setAttentionOutput] = useState<number[]>([])
  const [attentionWeights, setAttentionWeights] = useState<number[]>([])
  const [lstmTime, setLstmTime] = useState<number | null>(null)
  const [attentionTime, setAttentionTime] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const runInference = async () => {
    setLoading(true)
    setError('')

    const inputData = Array.from(
      { length: INPUT_SIZE * SEQUENCE_LENGTH },
      () => Math.random() - 0.5,
    )
    const sequence = Array.from(
      { length: SEQUENCE_LENGTH },
      (_, step) => inputData.slice(step * INPUT_SIZE, (step + 1) * INPUT_SIZE),
    )
    setInputSequence(sequence)

    try {
      const lstmResponse = await axios.post<LstmResponse>(
        '/api/predict/lstm',
        { input_data: inputData },
      )
      setLstmOutput(lstmResponse.data.output)
      setLstmTime(lstmResponse.data.execution_time_ms)

      const attentionResponse = await axios.post<AttentionResponse>(
        '/api/predict/attention',
        { input_data: inputData },
      )
      setAttentionOutput(attentionResponse.data.output)
      setAttentionWeights(attentionResponse.data.attention_weights)
      setAttentionTime(attentionResponse.data.execution_time_ms)
    } catch (requestError) {
      const message = axios.isAxiosError(requestError)
        ? String(requestError.response?.data?.detail ?? requestError.message)
        : 'Unknown inference error'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const hiddenChartData = lstmOutput.map((value, index) => ({
    position: `h${index}`,
    value,
  }))

  return (
    <main
      style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '24px',
        fontFamily: 'ui-monospace, SFMono-Regular, Consolas, monospace',
        color: '#172033',
      }}
    >
      <h1 style={{ marginBottom: '8px' }}>CUDA Sequence Inference Engine</h1>
      <p style={{ maxWidth: '900px', lineHeight: 1.6, color: '#4b5563' }}>
        This demonstration runs two hand-written sequence operations on an
        NVIDIA GPU. The single-layer LSTM carries memory through five input
        steps. The attention path lets the final step look back over all five
        steps. Fixed, seeded synthetic weights keep the demo reproducible; the
        goal is to demonstrate the engine and its performance, not a trained
        prediction.
      </p>

      <section
        style={{
          padding: '18px',
          margin: '22px 0',
          borderRadius: '10px',
          background: '#f3f6fa',
          lineHeight: 1.55,
        }}
      >
        <h2 style={{ marginTop: 0, fontSize: '20px' }}>What one click does</h2>
        <p>
          <strong>1.</strong> Generate five input vectors, each containing four
          numbers.
        </p>
        <p>
          <strong>2.</strong> Run the LSTM from left to right. cuBLAS calculates
          the input and recurrent projections; a custom CUDA kernel applies the
          forget, write, candidate, and output gates.
        </p>
        <p>
          <strong>3.</strong> Separately run last-query self-attention. The
          final input step receives five attention percentages—one for every
          step—and uses them to create an eight-number summary.
        </p>
      </section>

      <button
        onClick={runInference}
        disabled={loading}
        style={{
          padding: '11px 20px',
          border: 0,
          borderRadius: '6px',
          background: '#2563eb',
          color: 'white',
          fontSize: '16px',
          cursor: loading ? 'wait' : 'pointer',
          opacity: loading ? 0.7 : 1,
        }}
      >
        {loading ? 'Running on GPU…' : 'Run LSTM + Attention'}
      </button>

      {error && (
        <p
          role="alert"
          style={{
            padding: '12px',
            background: '#fee2e2',
            color: '#991b1b',
            borderRadius: '6px',
          }}
        >
          {error}
        </p>
      )}

      {inputSequence.length > 0 && (
        <section style={{ marginTop: '24px' }}>
          <h2 style={{ fontSize: '20px' }}>Generated input sequence</h2>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(5, minmax(120px, 1fr))',
              gap: '10px',
            }}
          >
            {inputSequence.map((step, stepIndex) => (
              <div
                key={stepIndex}
                style={{
                  padding: '10px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  background: 'white',
                }}
              >
                <strong>Step {stepIndex + 1}</strong>
                <div style={{ marginTop: '7px', color: '#4b5563' }}>
                  [{step.map((value) => value.toFixed(2)).join(', ')}]
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: '20px',
          marginTop: '24px',
        }}
      >
        <section
          style={{
            border: '1px solid #d1d5db',
            borderRadius: '8px',
            padding: '16px',
          }}
        >
          <h2 style={{ fontSize: '20px', marginTop: 0 }}>
            Final LSTM hidden state (h<sub>T</sub>)
          </h2>
          <p style={{ color: '#6b7280', minHeight: '42px' }}>
            These are the eight positions in the final hidden vector after all
            five steps—not five points in time.
          </p>
          {lstmOutput.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={hiddenChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="position" />
                  <YAxis />
                  <Tooltip formatter={(value: number) => value.toFixed(6)} />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#7c3aed"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
              <small>Compute time: {lstmTime?.toFixed(4)} ms</small>
            </>
          ) : (
            <p>No result yet.</p>
          )}
        </section>

        <section
          style={{
            border: '1px solid #d1d5db',
            borderRadius: '8px',
            padding: '16px',
          }}
        >
          <h2 style={{ fontSize: '20px', marginTop: 0 }}>
            Last-step attention weights
          </h2>
          <p style={{ color: '#6b7280', minHeight: '42px' }}>
            These five percentages add to 100%. A darker box contributed more
            to the attended summary.
          </p>
          {attentionWeights.length > 0 ? (
            <>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(5, 1fr)',
                  gap: '8px',
                  margin: '28px 0',
                }}
              >
                {attentionWeights.map((weight, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '16px 6px',
                      borderRadius: '6px',
                      textAlign: 'center',
                      background: `rgba(37, 99, 235, ${Math.min(1, 0.12 + weight * 2.2)})`,
                      color: weight > 0.3 ? 'white' : '#172033',
                    }}
                  >
                    <strong>Step {index + 1}</strong>
                    <div>{(weight * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>

              <h3 style={{ fontSize: '16px' }}>Attended summary</h3>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, 1fr)',
                  gap: '6px',
                }}
              >
                {attentionOutput.map((value, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '8px',
                      background: '#eef2ff',
                      borderRadius: '4px',
                      textAlign: 'center',
                    }}
                  >
                    v{index}: {value.toFixed(4)}
                  </div>
                ))}
              </div>
              <small style={{ display: 'block', marginTop: '12px' }}>
                Compute time: {attentionTime?.toFixed(4)} ms
              </small>
            </>
          ) : (
            <p>No result yet.</p>
          )}
        </section>
      </div>

      <BenchmarkChart />
    </main>
  )
}

export default App
