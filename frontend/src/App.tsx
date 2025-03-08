
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@/components/theme-provider';
import { Toaster } from "@/components/ui/toaster";
import Results from '@/pages/Results';
import NotFound from '@/pages/NotFound';
import Predict from '@/pages/Predict';
import './App.css';

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="landslide-theme">
      <Router>
        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/results" element={<Results />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Router>
      <Toaster />
    </ThemeProvider>
  );
}

export default App;
