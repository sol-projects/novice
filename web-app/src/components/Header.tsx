import React from "react";
import { NavLink, Link } from "react-router-dom";
import { FiFileText, FiMap, FiBarChart2, FiSettings } from "react-icons/fi";

export default function Header() {
  return (
    <header className="site-header">
      <div className="container header-inner">
        <Link to="/" className="wordmark" aria-label="Novice, domov"><svg className="kitty-mark" viewBox="0 0 90 72" aria-hidden="true"><path d="M13 29 12 6 33 17 Q45 12 57 17 L78 6 77 30 C92 66 65 69 45 69 S-2 64 13 29Z" fill="#fff" stroke="currentColor" strokeWidth="3"/><ellipse cx="31" cy="44" rx="3" ry="5" fill="currentColor"/><ellipse cx="59" cy="44" rx="3" ry="5" fill="currentColor"/><ellipse cx="45" cy="52" rx="4" ry="3" fill="#e0ad31"/><path d="m18 45-13-3m13 10L4 53m67-8 14-3m-14 10 15 1" fill="none" stroke="currentColor" strokeWidth="2"/><path d="M62 19Q50 0 48 16q-1 16 16 9Q81 37 83 23 87 8 65 15Z" fill="#d52765" stroke="currentColor" strokeWidth="2"/><circle cx="65" cy="20" r="5" fill="#ed5588" stroke="currentColor" strokeWidth="2"/></svg>novice<span aria-hidden="true">/</span></Link>
        <span className="site-description">Agregator in arhiv<br />slovenskih novic</span>
        <nav aria-label="Glavna navigacija">
          <NavLink to="/" end aria-label="Novice"><FiFileText aria-hidden="true" />Novice</NavLink>
          <NavLink to="/map" aria-label="Zemljevid"><FiMap aria-hidden="true" />Zemljevid</NavLink>
          <NavLink to="/chart" aria-label="Grafi"><FiBarChart2 aria-hidden="true" />Grafi</NavLink>
          <NavLink to="/about" aria-label="Nastavitve"><FiSettings aria-hidden="true" />Nastavitve</NavLink>
        </nav>
      </div>
    </header>
  );
}
