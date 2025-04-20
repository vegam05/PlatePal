import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  const location = useLocation();
  
  const isActive = (path) => {
    return location.pathname === path ? 'home' : '';
  };
  
  return (
    <div className="topnav">
      <h1 className="title">Plate Pal</h1>
      <nav className="navbar">
        <ul className="nav-list">
          <li className={`nav-link ${isActive('/')}`}>
            <Link to="/">Home</Link>
          </li>
          <li className={`nav-link ${isActive('/profile')}`}>
            <Link to="/profile">Health Profile</Link>
          </li>
          <li className="nav-link">
            <Link to="/news">News</Link>
          </li>
          <li className="nav-link">
            <Link to="/contact">Contact</Link>
          </li>
          <li className="nav-link">
            <Link to="/about">About</Link>
          </li>
        </ul>
      </nav>
    </div>
  );
}

export default Navbar;