import { NavLink, Route, Routes } from 'react-router-dom'
import { useUser } from './context/UserContext'
import ChatPage from './pages/ChatPage'
import MemoryBrowserPage from './pages/MemoryBrowserPage'
import RetrievalTracePage from './pages/RetrievalTracePage'
import StatsPage from './pages/StatsPage'
import ReflectionLogPage from './pages/ReflectionLogPage'
import ProceduralPage from './pages/ProceduralPage'

const NAV_ITEMS = [
  { to: '/', label: 'Chat', end: true },
  { to: '/memories', label: 'Memory Browser' },
  { to: '/trace', label: 'Retrieval Trace' },
  { to: '/stats', label: 'Timeline & Stats' },
  { to: '/reflection', label: 'Reflection Log' },
  { to: '/procedural', label: 'Procedural Strategies' },
]

function App() {
  const { userId, setUserId } = useUser()

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>mnemos</h1>
        <p className="tagline">persistent memory dashboard</p>
        <nav>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="user-switcher">
          <label htmlFor="user-id-input">User</label>
          <input
            id="user-id-input"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
        </div>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/memories" element={<MemoryBrowserPage />} />
          <Route path="/trace" element={<RetrievalTracePage />} />
          <Route path="/stats" element={<StatsPage />} />
          <Route path="/reflection" element={<ReflectionLogPage />} />
          <Route path="/procedural" element={<ProceduralPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
