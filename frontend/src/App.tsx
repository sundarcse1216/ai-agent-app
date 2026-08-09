import { ChatWindow } from "./components/ChatWindow";
import { useAgents } from "./hooks/useAgents";
import "./App.css";

function App() {
  const agents = useAgents();

  return (
    <div className="app">
      <header className="app__header">
        <h1>Smart AI Multi-Agent Assistant</h1>
        <p>Weather &middot; Database &middot; Recommendations &middot; Knowledge Search &middot; Image Studio &middot; Chat</p>
      </header>
      <main className="app__main">
        <ChatWindow agents={agents} />
      </main>
    </div>
  );
}

export default App;
