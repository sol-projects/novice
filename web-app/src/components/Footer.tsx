import React from "react";
import { FiGithub } from "react-icons/fi";

export default function Footer() {
  return (
    <footer className="container site-footer">
      <span>© {new Date().getFullYear()} SOL</span>
      <a href="https://github.com/sol-projects/novice" aria-label="GitHub"><FiGithub aria-hidden="true" />Izvorna koda</a>
    </footer>
  );
}
