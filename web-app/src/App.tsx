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
import Chart2 from "./components/Chart2";
import Chart3 from "./components/Chart3";
import Chart4 from "./components/Chart4";
import Map from "./components/Map";


export const App = () => (
  <ChakraProvider theme={theme}>
    <Router>
      <div>
        <Header />
        <Routes>
          <Route path="/" element={<News />} />
          <Route path="/map" element={<Map />} />
          <Route path="/chart" element={<Chart />} />
          <Route path="/chart2" element={<Chart2 />} />
          <Route path="/chart3" element={<Chart3 />} />
          <Route path="/chart4" element={<Chart4 />} />
          <Route path="/about" element={<News />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  </ChakraProvider>
);
