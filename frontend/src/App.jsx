import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import Body from './components/Body/Body';
import Footer from './components/Footer/Footer';
import HealthProfile from './components/HealthProfile/HealthProfile';
import './App.css';

const App = () => {
  return (
    <Router>
      <div className='site'>
        <div className='app'>
          <Navbar />
          <Routes>
            <Route path="/" element={<Body />} />
            <Route path="/profile" element={<HealthProfile />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </Router>
  );
};

export default App;