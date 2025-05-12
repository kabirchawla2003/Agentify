"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent } from "@/components/ui/card"
import { LucideSend, LucideLoader, LucideUser, LucideBot, LucideCoins } from "lucide-react"
import { saveQueryResult } from "@/app/actions"
import { cn } from "@/lib/utils"

type Message = {
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export function ChatInterface() {
  const [query, setQuery] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    const userMessage: Message = {
      role: "user",
      content: query,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)
    setQuery("")

    try {
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query.trim() }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response from API")
      }

      const data = await response.json()

      const assistantMessage: Message = {
        role: "assistant",
        content: data.response,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])

      // Save the result to history
      await saveQueryResult({
        query: userMessage.content,
        response: data.response,
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Error:", error)

      const errorMessage: Message = {
        role: "assistant",
        content: "Sorry, I encountered an error processing your request. Please try again.",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <Card className="flex-1 mb-4 overflow-hidden flex flex-col">
        <CardContent className="p-4 flex-1 overflow-y-auto max-h-[60vh]">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-8">
              <LucideCoins className="h-16 w-16 text-emerald-600 dark:text-emerald-400 mb-4" />
              <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-2">Financial Assistant</h2>
              <p className="text-slate-600 dark:text-slate-300 max-w-md">
                Ask me about stock analysis, loan approval, portfolio management, or any other financial questions.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={cn(
                    "flex gap-3 p-4 rounded-lg",
                    message.role === "user" ? "bg-slate-100 dark:bg-slate-700" : "bg-emerald-50 dark:bg-emerald-900/20",
                  )}
                >
                  <div className="flex-shrink-0">
                    {message.role === "user" ? (
                      <div className="w-8 h-8 rounded-full bg-slate-300 dark:bg-slate-600 flex items-center justify-center">
                        <LucideUser className="h-5 w-5 text-slate-700 dark:text-slate-300" />
                      </div>
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-emerald-200 dark:bg-emerald-800 flex items-center justify-center">
                        <LucideBot className="h-5 w-5 text-emerald-700 dark:text-emerald-300" />
                      </div>
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium mb-1">{message.role === "user" ? "You" : "Financial Assistant"}</div>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </CardContent>
      </Card>

      <form onSubmit={handleSubmit} className="relative">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about stocks, loans, portfolio analysis..."
          className="min-h-[100px] resize-none pr-12"
          disabled={isLoading}
        />
        <Button type="submit" size="icon" className="absolute bottom-3 right-3" disabled={isLoading || !query.trim()}>
          {isLoading ? <LucideLoader className="h-5 w-5 animate-spin" /> : <LucideSend className="h-5 w-5" />}
        </Button>
      </form>
    </div>
  )
}
