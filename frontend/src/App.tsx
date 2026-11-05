import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Message = {
  id: string
  role: 'user' | 'assistant'
  text: string
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const text = question.trim()
    if (!text || loading) return

    setError('')
    setLoading(true)
    setQuestion('')

    setMessages((previous) => [
      ...previous,
      { id: crypto.randomUUID(), role: 'user', text },
    ])

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
        signal: AbortSignal.timeout(100000),
      })

      if (!response.ok) {
        throw new Error(
          `Request failed (${response.status}). Check your backend terminals.`,
        )
      }

      const data = await response.json()

      if (typeof data.answer !== 'string' || !data.answer.trim()) {
        throw new Error('The server returned an empty answer.')
      }

      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: data.answer,
        },
      ])
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Unable to send your question.',
      )
      setQuestion(text)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <header className="header">
        <div className="brand-icon" aria-hidden="true">+</div>
        <div>
          <p className="eyebrow">YOUR BENEFITS, EXPLAINED</p>
          <h1>Healthcare Assistant</h1>
        </div>
      </header>

      <p className="intro">
        Ask about coverage, visit limits, and prior authorization.
      </p>

      <div className="demo-banner">
        Demo plan · Synthetic information · Each question is handled independently
      </div>

      <section
        className="conversation"
        aria-label="Conversation"
        aria-live="polite"
        aria-busy={loading}
      >
        {messages.length === 0 && (
          <div className="welcome">
            <h2>What would you like to understand?</h2>
            <p>Try: “Do I need approval before physical therapy?”</p>
          </div>
        )}

        {messages.map((message) => (
          <article
            key={message.id}
            className={`message ${message.role}`}
          >
            <strong>
              {message.role === 'user' ? 'You' : 'Healthcare assistant'}
            </strong>
            <p>{message.text}</p>
          </article>
        ))}

        {loading && (
          <p className="loading">Searching the demo plan and preparing an answer…</p>
        )}
      </section>

      {error && <p className="error" role="alert">{error}</p>}

      <form className="composer" onSubmit={sendMessage}>
        <label htmlFor="question">Your benefits question</label>
        <div className="input-row">
          <input
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about your demo plan…"
            maxLength={4000}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
          >
            {loading ? 'Waiting…' : 'Send'}
          </button>
        </div>
      </form>
    </main>
  )
}

export default App