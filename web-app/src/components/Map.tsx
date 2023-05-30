import { io } from "socket.io-client";
import React, { useState, useEffect } from "react";
import { Map, TileLayer } from "leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import INews from "../news/model";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
import markIcon from "../assets/marker.png"; //You can change market image here
import "leaflet.markercluster";
import "leaflet.markercluster/dist/MarkerCluster.css";
import "leaflet.markercluster/dist/MarkerCluster.Default.css";
import { VStack, Center } from "@chakra-ui/react";

const sloveniaBounds = [
  [45.4252, 13.3757],
  [46.8739, 16.6106],
];

const customIcon = L.icon({
  iconUrl: markIcon,
  iconSize: [42, 80],
  iconAnchor: [21, 80], //If you want to have marker centered you have to half values from iconSize
});

export default function MapComponent() {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);

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
        const { location, title, url } = article;
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
        const marker = L.marker(switchedCoordinates, { icon: customIcon });
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
  }, [filteredNews]);

  //v teoriji bi blo najboljše da je to <Center>zemljevid...</Center>
  //    to mi je delal neki cajta, ampak potem je random začelo crashat z JS napakami
  //    definitivno rabi bit dodan nek padding v CSS za umik od roba ekrana če chakra-ui komponente ne delajo s tem zemljevidom
  //height in width ne moreta bit v % (ne dela), ampak px ni v redu zared manjših ekranov
  //    to bi lahko v teoriji rešu tak da bi menjavu px glede na trenutno velikost zaslona, ne vem če bi to delal. mislim da ma chakra-ui neke stvari za to vgrajene tud? https://chakra-ui.com/docs/styled-system/responsive-styles

  //verjetno je problem to da editamo dom direktno z document.getElementById("map");
  //    to ne pomeni da je to narobe, verjetno ni pametno da dosti spreminjaš kodo
  return <div id="map" style={{ height: "600px", width: "1000px" }} />;
}
