import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, HashRouter } from 'react-router-dom'
import { STATIC_DEMO } from './api/client'
import './index.css'
import App from './App.tsx'
import { UserProvider } from './context/UserContext.tsx'

// GitHub Pages has no server-side routing, so the static demo uses hash URLs.
const Router = STATIC_DEMO ? HashRouter : BrowserRouter

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 5_000 } },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <UserProvider>
        <Router>
          <App />
        </Router>
      </UserProvider>
    </QueryClientProvider>
  </StrictMode>,
)
