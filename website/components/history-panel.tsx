"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LucideHistory, LucideTrash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { getQueryHistory, clearQueryHistory } from "@/app/actions"
import { ScrollArea } from "@/components/ui/scroll-area"

type HistoryItem = {
  id: string
  query: string
  response: string
  timestamp: string
}

export function HistoryPanel() {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [selectedItem, setSelectedItem] = useState<string | null>(null)

  const loadHistory = async () => {
    try {
      const data = await getQueryHistory()
      setHistory(data)
    } catch (error) {
      console.error("Error loading history:", error)
    }
  }

  useEffect(() => {
    loadHistory()
    const interval = setInterval(loadHistory, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleClearHistory = async () => {
    try {
      await clearQueryHistory()
      setHistory([])
      setSelectedItem(null)
    } catch (error) {
      console.error("Error clearing history:", error)
    }
  }

  return (
    <Card className="h-full w-full">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <LucideHistory className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
          Recent Queries
        </CardTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleClearHistory}
          disabled={history.length === 0}
          className="h-8 px-2"
        >
          <LucideTrash2 className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="p-4">
        {history.length === 0 ? (
          <div className="text-center py-8 text-slate-500 dark:text-slate-400">No query history yet</div>
        ) : (
          <ScrollArea className="h-[60vh] pr-4">
            <div className="space-y-3">
              {history.map((item) => (
                <div
                  key={item.id}
                  className={`p-3 rounded-md cursor-pointer transition-colors ${
                    selectedItem === item.id
                      ? "bg-emerald-100 dark:bg-emerald-900/30"
                      : "bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700"
                  }`}
                  onClick={() => setSelectedItem(selectedItem === item.id ? null : item.id)}
                >
                  <div className="font-medium text-sm truncate">{item.query}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    {new Date(item.timestamp).toLocaleString()}
                  </div>

                  {selectedItem === item.id && (
                    <div className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-700">
                      <div className="text-xs font-medium mb-1">Response:</div>
                      <div className="text-sm whitespace-pre-wrap max-h-40 overflow-y-auto">{item.response}</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  )
}
