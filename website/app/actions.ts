"use server"

import fs from "fs"
import path from "path"
import { v4 as uuidv4 } from "uuid"

const HISTORY_FILE = path.join(process.cwd(), "data", "query-history.json")

// Ensure the data directory exists
const ensureDataDir = () => {
  const dataDir = path.join(process.cwd(), "data")
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true })
  }

  if (!fs.existsSync(HISTORY_FILE)) {
    fs.writeFileSync(HISTORY_FILE, JSON.stringify([]))
  }
}

type QueryResult = {
  query: string
  response: string
  timestamp: string
}

export async function saveQueryResult(result: QueryResult) {
  ensureDataDir()

  // Read existing history
  let history = []
  try {
    const data = fs.readFileSync(HISTORY_FILE, "utf8")
    history = JSON.parse(data)
  } catch (error) {
    console.error("Error reading history file:", error)
    history = []
  }

  // Add new item with ID
  const newItem = {
    id: uuidv4(),
    ...result,
  }

  // Keep only the most recent 5 items
  history = [newItem, ...history].slice(0, 5)

  // Write back to file
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2))

  return { success: true }
}

export async function getQueryHistory() {
  ensureDataDir()

  try {
    const data = fs.readFileSync(HISTORY_FILE, "utf8")
    return JSON.parse(data)
  } catch (error) {
    console.error("Error reading history file:", error)
    return []
  }
}

export async function clearQueryHistory() {
  ensureDataDir()
  fs.writeFileSync(HISTORY_FILE, JSON.stringify([]))
  return { success: true }
}
