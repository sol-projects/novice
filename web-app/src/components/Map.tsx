import { io } from "socket.io-client";
import React, { useState, useEffect } from "react";
import { Map, TileLayer } from "leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import INews from "../news/model";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
//import "leaflet/dist/images/marker-icon-2x.png";
//import "leaflet/dist/images/marker-shadow.png";

const sloveniaBounds = [
  [45.4252, 13.3757],
  [46.8739, 16.6106],
];

const customIcon = L.icon({
  iconUrl: "path/to/custom/marker-icon.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
  shadowUrl: "path/to/custom/marker-shadow.png",
});

export default function MapComponent() {
  //This part of code is used to get all new fom db. It will get even new news when they are added. This is for loop  {filteredNews.map((article) => ( <ToDo> ))}
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
  //it ends here <3 <3 <3

  useEffect(() => {
    const mapContainer = document.getElementById("map");

    if (mapContainer && !("_leaflet_id" in mapContainer)) {
      const map = new Map("map", {
        center: [46.1512, 14.9955],
        zoom: 8
      });

      const tileLayer = new TileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "Map data © OpenStreetMap contributors",
        maxZoom: 19,
      });
      tileLayer.addTo(map);

    
      return () => {
        map.remove();
      };
    }
  }, []);

  return <div id="map" style={{ height: "400px" }} />;
}
