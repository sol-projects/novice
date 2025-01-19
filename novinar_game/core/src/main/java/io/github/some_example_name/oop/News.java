package io.github.some_example_name.oop;

import java.util.List;

public class News {
    private String title;
    private String content;
    private Location location;

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public Location getLocation() {
        return location;
    }

    public void setLocation(Location location) {
        this.location = location;
    }

    public static class Location {
        private String type; // e.g., "Point"
        private List<Double> coordinates; // [longitude, latitude]

        public String getType() {
            return type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public List<Double> getCoordinates() {
            return coordinates;
        }

        public void setCoordinates(List<Double> coordinates) {
            this.coordinates = coordinates;
        }

        public double getLongitude() {
            return coordinates.get(0); // Longitude
        }

        public double getLatitude() {
            return coordinates.get(1); // Latitude
        }
    }
}
