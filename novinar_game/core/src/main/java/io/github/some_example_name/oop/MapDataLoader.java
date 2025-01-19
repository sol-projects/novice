package io.github.some_example_name.oop;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.utils.viewport.Viewport;

import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MapDataLoader {

    public interface DataLoadedCallback {
        void onDataLoaded(List<MapMarker> markers);
    }

    public static void loadMarkers(int zoom, OrthographicCamera camera, Viewport viewport, DataLoadedCallback callback) {
        ApiService apiService = RetrofitClient.getClient().create(ApiService.class);

        Call<List<News>> call = apiService.getNews();
        call.enqueue(new Callback<List<News>>() {
            @Override
            public void onResponse(Call<List<News>> call, Response<List<News>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    List<MapMarker> markers = new ArrayList<>();
                    for (News news : response.body()) {
                        if (news.getLocation() != null && news.getLocation().getCoordinates() != null) {
                            double longitude = news.getLocation().getLongitude();
                            double latitude = news.getLocation().getLatitude();

                            // Convert geolocation to map positions
                            float x = calculateMapX(longitude, zoom);
                            float y = calculateMapY(latitude, zoom);

                            // Adjust for camera position and viewport
                            float adjustedX = x - (camera.position.x - viewport.getWorldWidth() / 2);
                            float adjustedY = y - (camera.position.y - viewport.getWorldHeight() / 2);

                            // Debug marker positions
                            System.out.println("Marker at: adjustedX = " + adjustedX + ", adjustedY = " + adjustedY);

                            // Schedule marker creation on the main thread
                            Gdx.app.postRunnable(() -> {
                                Texture texture = new Texture("marker.png");
                                markers.add(new MapMarker(adjustedX, adjustedY, texture));
                            });
                        }
                    }

                    // Notify the callback on the main thread
                    Gdx.app.postRunnable(() -> callback.onDataLoaded(markers));
                } else {
                    System.err.println("Failed to load news. Response code: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<List<News>> call, Throwable t) {
                t.printStackTrace();
            }
        });
    }

    private static float calculateMapX(double longitude, int zoom) {
        double tileCount = Math.pow(2, zoom);
        return (float) ((longitude + 180) / 360 * tileCount * 256);
    }

    private static float calculateMapY(double latitude, int zoom) {
        double tileCount = Math.pow(2, zoom);
        double radLat = Math.toRadians(latitude);
        return (float) ((1 - Math.log(Math.tan(radLat) + 1 / Math.cos(radLat)) / Math.PI) / 2 * tileCount * 256);
    }
}
