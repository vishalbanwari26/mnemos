import { createContext, useContext, useState, type ReactNode } from 'react'

interface UserContextValue {
  userId: string
  setUserId: (id: string) => void
}

const UserContext = createContext<UserContextValue | undefined>(undefined)

const STORAGE_KEY = 'mnemos.userId'

export function UserProvider({ children }: { children: ReactNode }) {
  const [userId, setUserIdState] = useState(
    () => localStorage.getItem(STORAGE_KEY) ?? 'demo-user',
  )

  function setUserId(id: string) {
    setUserIdState(id)
    localStorage.setItem(STORAGE_KEY, id)
  }

  return (
    <UserContext.Provider value={{ userId, setUserId }}>
      {children}
    </UserContext.Provider>
  )
}

export function useUser(): UserContextValue {
  const ctx = useContext(UserContext)
  if (!ctx) throw new Error('useUser must be used within UserProvider')
  return ctx
}
