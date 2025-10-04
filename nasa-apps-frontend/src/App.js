import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  Navigate,
} from "react-router-dom";
import "./App.css";
import ExoplanetInfo from "./pages/ExoplanetInfo";
import ExoplanetFinder from "./pages/ExoplanetFinder";

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link to="/" className="brand">
          NASA Apps
        </Link>
        <div className="nav-links">
          <Link to="/info" className="nav-link">
            What are Exoplanets?
          </Link>
          <Link to="/finder" className="nav-link primary">
            Exoplanet Finder
          </Link>
        </div>
      </div>
    </nav>
  );
}

function Home() {
  return (
    <div className="container">
      <section className="hero">
        <h1>Explore Exoplanets</h1>
        <p>
          Learn what exoplanets are and try a simple ML-powered finder using
          light curves.
        </p>
        <div className="hero-actions">
          <Link to="/info" className="button secondary">
            Learn
          </Link>
          <Link to="/finder" className="button">
            Try the Finder
          </Link>
        </div>
      </section>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/info" element={<ExoplanetInfo />} />
          <Route path="/finder" element={<ExoplanetFinder />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
