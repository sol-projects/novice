package io.github.some_example_name.oop;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class CoordinateExtractor {
    public static List<double[]> extractCoordinates(String jsonString) {
        List<double[]> coordinatesList = new ArrayList<>();

        try {
            JSONObject jsonObject = new JSONObject(jsonString);

            // Extract the "location" field
            JSONObject locationObject = jsonObject.getJSONObject("location");

            // Get the "coordinates" array
            JSONArray coordinatesArray = locationObject.getJSONArray("coordinates");

            // Extract longitude and latitude
            double longitude = coordinatesArray.getDouble(0);
            double latitude = coordinatesArray.getDouble(1);

            // Add to the list (as [longitude, latitude])
            coordinatesList.add(new double[]{longitude, latitude});

        } catch (Exception e) {
            e.printStackTrace();
        }

        return coordinatesList;
    }

    public static void main(String[] args) {
        // Example JSON string
        String jsonString = "{ \"location\": { \"type\": \"Point\", \"coordinates\": [12.345678, 98.765432] } }";

        // Extract coordinates
        List<double[]> coordinates = extractCoordinates(jsonString);

        // Print the extracted coordinates
        for (double[] coordinate : coordinates) {
            System.out.println("Longitude: " + coordinate[0] + ", Latitude: " + coordinate[1]);
        }
    }
}
