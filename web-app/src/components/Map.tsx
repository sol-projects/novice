import { io } from "socket.io-client";
import React, { useState, useEffect, useCallback } from "react";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import INews from "../news/model";
import { getAll } from "../news/api";
import { FilterData } from "./Filter";
import FilterPanel from "./FilterPanel";
import * as FilterFn from "../news/filter";
import weatherIcon from "../assets/weather.png"; //You can change market image here
import defoultIcon from "../assets/marker.png";
import sportIcon from "../assets/spotr.png";
import warIcon from "../assets/bomb.png";
import "leaflet.markercluster";
import "leaflet.markercluster/dist/MarkerCluster.css";
import "leaflet.markercluster/dist/MarkerCluster.Default.css";
import { FiPlay } from "react-icons/fi";

const customIcon = L.icon({
  iconUrl: defoultIcon,
  iconSize: [42, 80],
  iconAnchor: [21, 80],
});

const customIconWeather = L.icon({
  iconUrl: weatherIcon,
  iconSize: [42, 42],
  iconAnchor: [21, 21], //If you want to have marker centered you have to half values from iconSize
});

const customIconWar = L.icon({
  iconUrl: warIcon,
  iconSize: [42, 42],
  iconAnchor: [21, 80],
});

const customIconSport = L.icon({
  iconUrl: sportIcon,
  iconSize: [42, 42],
  iconAnchor: [21, 21],
});

export default function MapComponent() {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [code, setCode] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [running, setRunning] = useState(false);
  const [codeError, setCodeError] = useState("");

  const handleRunCode = async () => {
    setRunning(true);
    setCodeError("");
    try {
      const response = await fetch("http://localhost:8000/news/geolang", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ code }),
      });

      if (!response.ok) {
        throw new Error("Error executing geolang request");
      }

      const data = await response.json();
      setOutput(JSON.stringify(data));
    } catch (error) {
      console.error(error);
      setCodeError("Kode ni bilo mogoče izvesti. Preveri kodo in povezavo s strežnikom.");
    } finally {
      setRunning(false);
    }
  };

  const handleCodeChange = (event: any) => {
    setCode(event.target.value);
  };

  useEffect(() => {
    const fetchData = async () => {
      const data = await getAll();
      if (data) {
        setNews(data);
        setFilteredNews(data);
      }
      setFailed(!data);
      setLoading(false);
    };

    fetchData();
  }, []);

  useEffect(() => {
    const socket = io("ws://localhost:8000/news");
    socket.on("news-added", (newNews) => {
      setNews(newNews);
      setFilteredNews(newNews);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const handleFilterChange = useCallback((filterData: FilterData) => {
    let filtered = news;
    filterData.categories = filterData.categories.filter((item) => item !== "");
    filterData.authors = filterData.authors.filter((item) => item !== "");
    if (filterData.categories.length > 0) {
      filtered = FilterFn.categories(filtered, filterData.categories);
    }

    if (filterData.authors.length > 0) {
      filtered = FilterFn.authors(filtered, filterData.authors);
    }

    if(filterData.websites.length > 0) {
      filtered = FilterFn.websites(filtered, filterData.websites);
    }

    if (filterData.title.length > 0) {
      filtered = FilterFn.title(filtered, filterData.title);
    }

    if (filterData.content.length > 0) {
      filtered = FilterFn.content(filtered, filterData.content);
    }

    filtered = FilterFn.date(filtered, filterData.from, filterData.to);
    setFilteredNews(filtered);
  }, [news]);

  useEffect(() => {
    const mapContainer = document.getElementById("map");

    if (mapContainer && !("_leaflet_id" in mapContainer)) {
      const map = new L.Map("map", {
        center: [46.1512, 14.9955],
        zoom: 8,
      });

      if (output) {
        const parsedOutput: any = JSON.parse(output);

        parsedOutput.features.forEach((feature: any) => {
          const { geometry, properties } = feature;

          if (geometry.type === "Point") {
            const { coordinates } = geometry;
            const { title } = properties;

            L.marker([coordinates[1], coordinates[0]], { icon: customIcon }).addTo(map).bindPopup(title);
          } else if (geometry.type === "Polygon") {
            const { coordinates } = geometry;
            const { title } = properties;

            const latLngs = coordinates[0].map((coords: any) => [
              coords[1],
              coords[0],
            ]);
            L.polygon(latLngs).addTo(map).bindPopup(title);
          } else if (geometry.type === "LineString") {
            const { coordinates } = geometry;
            const { title } = properties;

            const latLngs = coordinates.map((coords: any) => [
              coords[1],
              coords[0],
            ]);
            L.polyline(latLngs).addTo(map).bindPopup(title);
          }
        });
      }
      const tileLayer = new L.TileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
          attribution: "Map data © OpenStreetMap contributors",
          maxZoom: 19,
        }
      );
      tileLayer.addTo(map);

      const markerClusterGroup = L.markerClusterGroup();

      filteredNews.forEach((article) => {
        const { location, title, url, categories } = article;
        const { coordinates } = location;

        if (
          coordinates.length < 2 ||
          coordinates[0] === 0 ||
          coordinates[1] === 0
        ) {
          return;
        }

        const switchedCoordinates: L.LatLngTuple = [
          coordinates[1],
          coordinates[0],
        ];

        let markerIcon = customIcon;

        if (
          categories.some((category) =>
            [
              "toča",
              "nevihta",
              "vreme",
              "dež",
              "megla",
              "sončno",
              "sneg",
              "sneženo",
              "ploha",
            ].includes(category)
          )
        ) {
          markerIcon = customIconWeather;
        } else if (
          categories.some((category) =>
            [
              "rekreacija",
              "gibanje",
              "šport",
              "nogomet",
              "košarka",
              "sport",
              "tenis",
              "gimnastika",
              "jahanje",
              "smučanje",
              "smuk",
              "rokomet",
            ].includes(category)
          )
        ) {
          markerIcon = customIconSport;
        } else if (
          categories.some((category) =>
            ["vojna", "napad", "bombandiranje", "tank"].includes(category)
          )
        ) {
          markerIcon = customIconWar;
        }

        const marker = L.marker(switchedCoordinates, { icon: markerIcon });
        const popup = document.createElement("div");
        const heading = document.createElement("strong");
        heading.textContent = title;
        const link = document.createElement("a");
        link.textContent = url;
        if (/^https?:\/\//i.test(url)) link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        popup.append(heading, document.createElement("br"), link);
        marker.bindPopup(popup);

        markerClusterGroup.addLayer(marker);
      });
      const updateMapSize = () => {
        map.invalidateSize();
      };
      map.addLayer(markerClusterGroup);
      const resizeObserver = new ResizeObserver(updateMapSize);
      resizeObserver.observe(mapContainer);
      return () => {
        map.remove();
        resizeObserver.disconnect();
      };
    }
  }, [filteredNews, output]);


  return (
    <>
      <h1 className="visually-hidden">Zemljevid novic</h1>
      <div className="filter-layout">
        <FilterPanel onChange={handleFilterChange} />
        <div className="map-workspace filter-content">
        <section className="map-region" aria-label="Zemljevid novic">
          <div className="workspace-label"><h2>Zemljevid</h2><span role="status">{loading ? "Nalaganje…" : failed ? "Strežnik ni dosegljiv" : `${filteredNews.length} novic v izboru`}</span></div>
          {failed && <p className="inline-error" role="alert">Novic ni bilo mogoče naložiti. Preveri povezavo in <button className="text-button" onClick={() => window.location.reload()}>poskusi znova</button>.</p>}
          <div id="map" />
          <div className="map-legend" aria-label="Legenda zemljevida">
            {[[defoultIcon, "Novice"], [weatherIcon, "Vreme"], [sportIcon, "Šport"], [warIcon, "Vojna"]].map(([icon, label]) => <span key={label}><img src={icon} alt="" />{label}</span>)}
          </div>
        </section>
        <aside className="code-panel" aria-labelledby="geolang-heading">
          <h2 id="geolang-heading">GeoLang</h2><p>Programska koda za prikaz geografskih podatkov.</p>
          <label htmlFor="geolang-code">Urejevalnik kode</label>
          <textarea id="geolang-code" value={code} onChange={handleCodeChange} placeholder="geolang programska koda" spellCheck={false} />
          <button className="button" onClick={handleRunCode} disabled={running || !code.trim()} aria-busy={running}><FiPlay aria-hidden="true" />{running ? "Izvajanje…" : "Zaženi kodo"}</button>
          {codeError && <p className="inline-error" role="alert">{codeError}</p>}
          {output && <details className="code-output"><summary>Rezultat</summary><pre>{output}</pre></details>}
        </aside>
        </div>
      </div>
    </>
  );
}
