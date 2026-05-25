import type { SSEEvent } from '@/types'

export function streamSSE(
  url: string,
  onEvent: (e: SSEEvent) => void,
  onError: (err: string) => void,
): () => void {
  const es = new EventSource(url)
  es.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data) as SSEEvent
      onEvent(data)
      if (data.event === 'done') es.close()
    } catch (err) {
      onError(`bad SSE payload: ${String(err)}`)
      es.close()
    }
  }
  es.onerror = () => {
    // EventSource auto-reconnects, which we don't want once 'done' has fired.
    if (es.readyState === EventSource.CLOSED) return
    onError('stream connection error')
    es.close()
  }
  return () => es.close()
}
