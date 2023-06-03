import { io } from "socket.io-client";
import React, { useState, useEffect } from "react";
import { Map, TileLayer } from "leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import INews from "../news/model";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
import weatherIcon from "../assets/weather.png"; //You can change market image here
import defoultIcon from "../assets/marker.png";
import sportIcon from "../assets/spotr.png";
import warIcon from "../assets/bomb.png";
import "leaflet.markercluster";
import "leaflet.markercluster/dist/MarkerCluster.css";
import "leaflet.markercluster/dist/MarkerCluster.Default.css";
import { VStack, HStack, Center, Textarea, Button, Box } from "@chakra-ui/react";
import { geolang } from '../news/geolang';

const sloveniaBounds = [
  [45.4252, 13.3757],
  [46.8739, 16.6106],
];

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
  const [code, setCode] = useState('');
  const [output, setOutput] = useState('');

  const handleRunCode = async () => {
    try {
      const response = await fetch('http://localhost:8000/news/geolang', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });

      if (!response.ok) {
        throw new Error('Error executing geolang request');
      }

      const data = await response.json();
      setOutput(JSON.stringify(data));
    } catch (error) {
      console.error(error);
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

  const handleFilterChange = (filterData: FilterData) => {
    let filtered = news;
    filterData.categories = filterData.categories.filter((item) => item !== "");
    filterData.authors = filterData.authors.filter((item) => item !== "");
    if (filterData.categories.length > 0) {
      filtered = FilterFn.categories(filtered, filterData.categories);
    }

    if (filterData.authors.length > 0) {
      filtered = FilterFn.authors(filtered, filterData.authors);
    }

    setFilteredNews(filtered);
  };

  useEffect(() => {
    const mapContainer = document.getElementById("map");

    if (mapContainer && !("_leaflet_id" in mapContainer)) {
      const map = new L.Map("map", {
        center: [46.1512, 14.9955],
        zoom: 8,
      });

      if (output) {
        const parsedOutput: any = JSON.parse(output); // Parse the output as JSON if needed

        parsedOutput.features.forEach((feature: any) => {
          const { geometry, properties } = feature;

          if (geometry.type === 'Point') {
            const { coordinates } = geometry;
            const { title } = properties;

            L.marker(coordinates.reverse())
              .addTo(map)
              .bindPopup(title);
          } else if (geometry.type === 'Polygon') {
            const { coordinates } = geometry;
            const { title } = properties;

            const latLngs = coordinates[0].map((coords: any) => [coords[1], coords[0]]);
            L.polygon(latLngs)
              .addTo(map)
              .bindPopup(title);
          } else if (geometry.type === 'LineString') {
            const { coordinates } = geometry;
            const { title } = properties;

            const latLngs = coordinates.map((coords: any) => [coords[1], coords[0]]);
            L.polyline(latLngs)
              .addTo(map)
              .bindPopup(title);
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
        const { type, coordinates } = location;

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

        categories.forEach((category) => {
          if (category.toLowerCase() === "toča" || category.toLowerCase() === "nevihta" || category.toLowerCase() === "vreme" || category.toLowerCase() === "dež") {
            markerIcon = customIconWeather;
          } else if (category.toLowerCase() === "sport" || category.toLowerCase() === "šport" || category.toLowerCase() === "nogomet" || category.toLowerCase() === "košarka") {
            markerIcon = customIconSport;
          } else if (category.toLowerCase() === "vojna" || category.toLowerCase() === "napad") {
            markerIcon = customIconWar;
          }
        });


        const marker = L.marker(switchedCoordinates, { icon: markerIcon });
        marker.bindPopup(
          `<b>${title}</b><br><a href="${url}" target="_blank">${url}</a>`
        );

        markerClusterGroup.addLayer(marker);
      });

      map.addLayer(markerClusterGroup);
      return () => {
        map.remove();
      };
    }
  }, [filteredNews, output]);

  //v teoriji bi blo najboljše da je to <Center>zemljevid...</Center>
  //    to mi je delal neki cajta, ampak potem je random začelo crashat z JS napakami
  //    definitivno rabi bit dodan nek padding v CSS za umik od roba ekrana če chakra-ui komponente ne delajo s tem zemljevidom
  //height in width ne moreta bit v % (ne dela), ampak px ni v redu zared manjših ekranov
  //    to bi lahko v teoriji rešu tak da bi menjavu px glede na trenutno velikost zaslona, ne vem če bi to delal. mislim da ma chakra-ui neke stvari za to vgrajene tud? https://chakra-ui.com/docs/styled-system/responsive-styles

  //verjetno je problem to da editamo dom direktno z document.getElementById("map");
  //    to ne pomeni da je to narobe, verjetno ni pametno da dosti spreminjaš kodo

  return <HStack>
      <Box>
        <div id="map" style={{ height: "600px", width: "1000px" }} />


      </Box>
    <VStack>
      <Textarea
        value={code}
        onChange={handleCodeChange}
        placeholder="geolang programska koda"
        height="500px"
        resize="none"
      />
      <Button colorScheme="blue" onClick={handleRunCode} mt={4}>
        Zaženi kodo
      </Button>
        <HStack spacing={2}>
          <Box>
            <img src={defoultIcon} alt="Default Icon" style={{ width: "24px", height: "24px" }} />
            <span>Default Icon</span>
          </Box>
          <Box>
            <img src={weatherIcon} alt="Weather Icon" style={{ width: "24px", height: "24px" }} />
            <span>Weather Icon</span>
          </Box>
          <Box>
            <img src={sportIcon} alt="Sport Icon" style={{ width: "24px", height: "24px" }} />
            <span>Sport Icon</span>
          </Box>
          <Box>
            <img src={warIcon} alt="War Icon" style={{ width: "24px", height: "24px" }} />
            <span>War Icon</span>
          </Box>
        </HStack>
    </VStack>
  </HStack>;
}
