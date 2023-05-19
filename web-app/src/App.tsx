import * as React from "react";
import {
  ChakraProvider,
  Box,
  Text,
  Link,
  VStack,
  Code,
  Grid,
  theme,
} from "@chakra-ui/react";
import { ColorModeSwitcher } from "./ColorModeSwitcher";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Header from "./components/Header";
import Footer from "./components/Footer";
import News from "./components/News";
import Chart from "./components/Chart";

export const App = () => (
  <ChakraProvider theme={theme}>
    <Router>
      <div>
        <Header />
        <Routes>
          <Route path="/" element={<News />} />
          <Route path="/map" element={<News />} />
          <Route path="/chart" element={<Chart />} />
          <Route path="/about" element={<News />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  </ChakraProvider>
);
