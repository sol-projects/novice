package io.github.some_example_name.oop;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import java.util.ArrayList;
import java.util.List;

public class ApiClient {
    private static final String API_URL = "http://localhost:8000/news/";

    public static List<double[]> fetchLocations() {
        List<double[]> locations = new ArrayList<>();
        OkHttpClient client = new OkHttpClient();

        Request request = new Request.Builder()
            .url(API_URL)
            .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() && response.body() != null) {
                String responseBody = response.body().string();
                ObjectMapper objectMapper = new ObjectMapper();
                JsonNode root = objectMapper.readTree(responseBody);

                for (JsonNode node : root) {
                    JsonNode location = node.get("location");
                    if (location != null && location.get("coordinates") != null) {
                        JsonNode coordinates = location.get("coordinates");
                        double lon = coordinates.get(0).asDouble();
                        double lat = coordinates.get(1).asDouble();
                        locations.add(new double[]{lon, lat});
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return locations;
    }
}
