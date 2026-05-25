import { useState, type KeyboardEvent } from 'react'
import type { DemoQuestion } from '@/types'
import { ArrowUp, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Props {
  onSubmit: (q: string) => void
  isStreaming: boolean
  demos: DemoQuestion[]
}

export function QueryBar({ onSubmit, isStreaming, demos }: Props) {
  const [value, setValue] = useState('')
  const [showDemos, setShowDemos] = useState(false)

  const submit = () => {
    const q = value.trim()
    if (!q || isStreaming) return
    onSubmit(q)
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const pickDemo = (q: DemoQuestion) => {
    setValue(q.question)
    setShowDemos(false)
    onSubmit(q.question)
  }

  return (
    <div className="w-full">
      <div className="relative">
        <div
          className={cn(
            'flex items-end gap-2 rounded-3xl border border-neutral-200 bg-white px-4 py-3',
            'shadow-sm transition-shadow focus-within:shadow-md',
          )}
        >
          <textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ask anything about the wiki..."
            rows={1}
            className="flex-1 resize-none bg-transparent text-[15px] leading-6 outline-none placeholder:text-neutral-400"
            style={{ maxHeight: '160px' }}
          />
          <button
            onClick={submit}
            disabled={!value.trim() || isStreaming}
            className={cn(
              'flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors',
              value.trim() && !isStreaming
                ? 'bg-neutral-900 text-white hover:bg-neutral-700'
                : 'bg-neutral-100 text-neutral-300',
            )}
            aria-label="Send"
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="mt-2 flex items-center justify-between px-1">
        <button
          onClick={() => setShowDemos(!showDemos)}
          disabled={isStreaming || demos.length === 0}
          className="flex items-center gap-1 text-xs text-neutral-500 hover:text-neutral-800 disabled:opacity-40"
        >
          Try a benchmark question
          <ChevronDown
            className={cn('h-3 w-3 transition-transform', showDemos && 'rotate-180')}
          />
        </button>
        <span className="text-[11px] text-neutral-400">⏎ to send · ⇧⏎ for newline</span>
      </div>

      {showDemos && (
        <div className="mt-2 overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-sm">
          {demos.map((q) => (
            <button
              key={q.id}
              onClick={() => pickDemo(q)}
              className="block w-full border-b border-neutral-100 px-4 py-2.5 text-left text-sm text-neutral-800 last:border-b-0 hover:bg-neutral-50"
            >
              <div className="flex items-start gap-2">
                <span
                  className={cn(
                    'mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider',
                    q.type === 'bridge'
                      ? 'bg-emerald-50 text-emerald-700'
                      : 'bg-blue-50 text-blue-700',
                  )}
                >
                  {q.type[0]}/{q.level[0]}
                </span>
                <span className="flex-1">{q.question}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
