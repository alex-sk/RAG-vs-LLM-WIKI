import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { MitigationsBanner } from '@/components/MitigationsBanner'
import { PipelinePanel } from '@/components/PipelinePanel'
import { QueryBar } from '@/components/QueryBar'
import { streamSSE } from '@/lib/sse'
import {
  emptyPipelineState,
  type DemoQuestion,
  type PipelineState,
  type RerankMode,
  type Verdict as VerdictResult,
} from '@/types'
import { Check, Loader2, X } from 'lucide-react'

export default function App() {
  const [rag, setRag] = useState<PipelineState>(emptyPipelineState())
  const [wiki, setWiki] = useState<PipelineState>(emptyPipelineState())
  const [demos, setDemos] = useState<DemoQuestion[]>([])
  const [currentQ, setCurrentQ] = useState<string>('')
  const [rerankMode, setRerankMode] = useState<RerankMode>('none')
  const cancellers = useRef<Array<() => void>>([])

  useEffect(() => {
    fetch('/api/demo-questions')
      .then((r) => r.json())
      .then(setDemos)
      .catch(() => setDemos([]))
  }, [])

  const isStreaming = rag.status === 'streaming' || wiki.status === 'streaming'

  const gold = useMemo(() => {
    return demos.find((d) => d.question === currentQ)?.answer
  }, [demos, currentQ])

  const runJudge = (
    question: string,
    gold: string,
    answer: string,
    set: Dispatch<SetStateAction<PipelineState>>,
  ) => {
    const ctrl = new AbortController()
    cancellers.current.push(() => ctrl.abort())
    fetch('/api/judge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, gold, answer }),
      signal: ctrl.signal,
    })
      .then((r) => r.json())
      .then((verdict: VerdictResult) =>
        set((s) => ({ ...s, judging: false, verdict })),
      )
      .catch((err) => {
        if (err.name === 'AbortError') return
        set((s) => ({ ...s, judging: false, verdict: null }))
      })
  }

  const runQuery = (question: string) => {
    cancellers.current.forEach((c) => c())
    cancellers.current = []
    setCurrentQ(question)
    setRag({ ...emptyPipelineState(), status: 'streaming' })
    setWiki({ ...emptyPipelineState(), status: 'streaming' })

    const goldForQuery = demos.find((d) => d.question === question)?.answer

    const ragUrl = `/api/rag?question=${encodeURIComponent(question)}&rerank=${rerankMode}`
    const wikiUrl = `/api/wiki?question=${encodeURIComponent(question)}`

    const accum = { rag: '', wiki: '' }
    const setters: Array<{
      url: string
      set: Dispatch<SetStateAction<PipelineState>>
      key: 'rag' | 'wiki'
    }> = [
      { url: ragUrl, set: setRag, key: 'rag' },
      { url: wikiUrl, set: setWiki, key: 'wiki' },
    ]

    for (const { url, set, key } of setters) {
      const cancel = streamSSE(
        url,
        (ev) => {
          if (ev.event === 'retrieved_chunks') {
            set((s) => ({ ...s, chunks: ev.chunks }))
          } else if (ev.event === 'reranked_chunks') {
            set((s) => ({ ...s, rerankedChunks: ev.chunks }))
          } else if (ev.event === 'tool_call' || ev.event === 'tool_result') {
            set((s) => ({ ...s, toolEvents: [...s.toolEvents, ev] }))
          } else if (ev.event === 'token') {
            accum[key] += ev.text
            set((s) => ({ ...s, answer: s.answer + ev.text }))
          } else if (ev.event === 'done') {
            const { event: _e, ...metrics } = ev
            const willJudge = !!goldForQuery && !!accum[key]
            set((s) => ({ ...s, status: 'done', metrics, judging: willJudge }))
            if (willJudge) {
              runJudge(question, goldForQuery!, accum[key], set)
            }
          }
        },
        (err) => set((s) => ({ ...s, status: 'error', error: err })),
      )
      cancellers.current.push(cancel)
    }
  }

  const ragSubtitle =
    rerankMode === 'none'
      ? 'embed → top-5 chunks → gpt-4o'
      : rerankMode === 'cross-encoder'
        ? 'embed → top-20 → cross-encoder ↓ top-5 → gpt-4o'
        : 'embed → top-20 → llm rerank ↓ top-5 → gpt-4o'

  return (
    <div className="flex h-full flex-col bg-neutral-50">
      <header className="border-b border-neutral-200 bg-white">
        <div className="mx-auto max-w-6xl px-6 py-4">
          <h1 className="text-base font-semibold tracking-tight text-neutral-900">
            RAG <span className="text-neutral-400">vs</span> LLM Wiki
          </h1>
          <p className="mt-0.5 text-xs text-neutral-500">
            Same model · same corpus · different retrieval
          </p>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-5 px-6 py-6">
        <QueryBar onSubmit={runQuery} isStreaming={isStreaming} demos={demos} />

        <MitigationsBanner />

        {gold && (rag.status === 'done' || wiki.status === 'done') && (
          <div className="flex items-center gap-3 rounded-xl border border-neutral-200 bg-white px-4 py-2.5 text-xs">
            <span className="text-neutral-500">Ground truth (HotpotQA):</span>
            <span className="font-mono font-medium text-neutral-900">{gold}</span>
            <span className="ml-auto flex items-center gap-3">
              <Verdict label="RAG" state={rag} />
              <Verdict label="Wiki" state={wiki} />
            </span>
          </div>
        )}

        <div className="grid flex-1 grid-cols-2 gap-5">
          <PipelinePanel
            title="RAG"
            subtitle={ragSubtitle}
            accent="rag"
            state={rag}
            rerankMode={rerankMode}
            onRerankChange={setRerankMode}
            disableRerankControl={isStreaming}
          />
          <PipelinePanel
            title="LLM Wiki"
            subtitle="agent · glob / read_file / grep → gpt-4o"
            accent="wiki"
            state={wiki}
          />
        </div>
      </main>
    </div>
  )
}

function Verdict({ label, state }: { label: string; state: PipelineState }) {
  if (state.status !== 'done') return null
  return (
    <span className="inline-flex items-center gap-1 font-medium">
      <span className="text-neutral-600">{label}</span>
      {state.judging ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin text-neutral-400" />
      ) : state.verdict ? (
        <span title={state.verdict.reason} className="inline-flex">
          {state.verdict.correct ? (
            <Check className="h-3.5 w-3.5 text-emerald-600" />
          ) : (
            <X className="h-3.5 w-3.5 text-red-500" />
          )}
        </span>
      ) : null}
    </span>
  )
}
