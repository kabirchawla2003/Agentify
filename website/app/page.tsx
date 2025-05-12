import { ChatInterface } from "@/components/chat-interface"
import { Header } from "@/components/header"
import { HistoryPanel } from "@/components/history-panel"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <Header />
      <div className="container mx-auto px-4 py-8 flex-1 flex flex-col md:flex-row gap-6">
        <div className="flex-1">
          <ChatInterface />
        </div>
        <div className="w-full md:w-96">
          <HistoryPanel />
        </div>
      </div>
    </main>
  )
}
