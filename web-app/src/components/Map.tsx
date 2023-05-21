import React, { useEffect } from "react";
import { Map, TileLayer } from "leaflet";
import "leaflet/dist/leaflet.css";

import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet/dist/images/marker-icon-2x.png";
import "leaflet/dist/images/marker-shadow.png";

const sloveniaBounds = [
  [45.4252, 13.3757], // Southwest boundary coordinates of Slovenia
  [46.8739, 16.6106], // Northeast boundary coordinates of Slovenia
];

// Custom marker icon
const customIcon = L.icon({
  iconUrl: "path/to/custom/marker-icon.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
  shadowUrl: "path/to/custom/marker-shadow.png",
});

export default function MapComponent() {
  useEffect(() => {
    const mapContainer = document.getElementById("map");

    if (mapContainer && !('_leaflet_id' in mapContainer)) {
      // Create the map instance
    const map = new Map("map", {
      center: [46.1512, 14.9955], // Center coordinates of Slovenia
      zoom: 8,
      maxBounds: [
        [46.3279, 13.3757], // Southwestern boundary coordinates of Slovenia
        [46.8634, 16.6106], // Northeastern boundary coordinates of Slovenia
      ],
    });

    // Add the tile layer
    const tileLayer = new TileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        attribution: "Map data © OpenStreetMap contributors",
        maxZoom: 19,
      }
    );
    tileLayer.addTo(map);

    // Set the view to Slovenia's bounds
    //map.fitBounds(sloveniaBounds);

    // Add a custom marker to Slovenia's center
    const marker = L.marker([46.1512, 14.9955], { icon: customIcon }).addTo(map);
    marker.bindPopup("Slovenia").openPopup();
  }
}, []);

return <div id="map" style={{ height: "400px" }} />;
}
