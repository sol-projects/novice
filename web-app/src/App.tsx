import * as React from "react";
import { ChakraProvider, extendTheme } from "@chakra-ui/react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import News from "./components/News";
import Chart from "./components/Chart";
import Map from "./components/Map";
import Settings from "./components/Settings";
import { AppearanceProvider } from "./appearance";
import "./styles.css";

const theme = extendTheme({
  config: { initialColorMode: "light", useSystemColorMode: false },
  fonts: { heading: "var(--font-display)", body: "var(--font-body)" },
  styles: { global: { body: { bg: "var(--color-paper)", color: "var(--color-ink)" } } },
});

export const App = () => (
  <ChakraProvider theme={theme}>
    <AppearanceProvider>
    <Router>
      <div className="app-shell">
        <a className="skip-link" href="#main">Preskoči na vsebino</a>
        <Header />
        <main id="main" className="container">
          <Routes>
            <Route path="/" element={<News />} />
            <Route path="/map" element={<Map />} />
            <Route path="/chart" element={<Chart />} />
            <Route path="/about" element={<Settings />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
    </AppearanceProvider>
  </ChakraProvider>
);
