import { LucideCoins, LucideLineChart } from "lucide-react"

export function Header() {
  return (
    <header className="bg-white dark:bg-slate-800 shadow-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <LucideCoins className="h-8 w-8 text-emerald-600 dark:text-emerald-400" />
            <h1 className="text-2xl font-bold text-slate-800 dark:text-white">Financial Assistant</h1>
          </div>
          <div className="flex items-center gap-2">
            <LucideLineChart className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
            <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
              Powered by AI Financial Tools
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}
