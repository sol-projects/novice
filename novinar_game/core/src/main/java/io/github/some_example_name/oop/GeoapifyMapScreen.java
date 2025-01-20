package io.github.some_example_name.oop;

import com.badlogic.gdx.Game;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.utils.viewport.FitViewport;
import com.badlogic.gdx.utils.viewport.Viewport;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashSet;
import java.util.List;

public class GeoapifyMapScreen implements Screen {

    private SpriteBatch batch;
    private Viewport viewport;
    private Texture staticMapTexture;

    private static final String GEOAPIFY_API_KEY = "ff9e520ad9624811884a79d85ecf6269";

    private double centerLon = 14.505751;
    private double centerLat = 46.056946;
    private double zoom = 4.3497;

    private List<double[]> markers;

    private final Game game;
    private final MainGameScreen mainGameScreen;

    public GeoapifyMapScreen(Game game, MainGameScreen mainGameScreen) {
        this.game = game;
        this.mainGameScreen = mainGameScreen;
        viewport = new FitViewport(800, 600);
        batch = new SpriteBatch();
        markers = ApiClient.fetchLocations(); // Fetch dynamic markers from API
        staticMapTexture = fetchStaticMap();
    }

    private static final int MAX_MARKERS_PER_REQUEST = 10;

    private String buildStaticMapUrl() {
        StringBuilder markerParams = new StringBuilder();
        HashSet<String> uniqueLocations = new HashSet<>();

        for (double[] marker : markers) {
            String locationKey = String.format("%.6f,%.6f", marker[0], marker[1]);

            if (!uniqueLocations.contains(locationKey)) {
                uniqueLocations.add(locationKey);
                markerParams.append(String.format(
                    "&marker=lonlat%%3A%.6f%%2C%.6f%%3Btype%%3Aawesome%%3Bcolor%%3A%%23bb3f73%%3Bsize%%3Asmall%%3Bicon%%3Apin",
                    marker[0], marker[1]
                ));
            }
        }

        return String.format(
            "https://maps.geoapify.com/v1/staticmap?style=osm-bright-smooth&width=800&height=600&center=lonlat%%3A%.6f%%2C%.6f&zoom=%.2f%s&apiKey=%s",
            centerLon, centerLat, zoom, markerParams.toString(), GEOAPIFY_API_KEY
        );
    }

    private Texture fetchStaticMap() {
        try {
            String url = buildStaticMapUrl();
            URL mapUrl = new URL(url);
            HttpURLConnection connection = (HttpURLConnection) mapUrl.openConnection();
            connection.setRequestMethod("GET");
            connection.connect();

            if (connection.getResponseCode() == 200) {
                InputStream inputStream = connection.getInputStream();
                FileOutputStream fileOutputStream = new FileOutputStream("static_map.png");

                byte[] buffer = new byte[1024];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    fileOutputStream.write(buffer, 0, bytesRead);
                }

                fileOutputStream.close();
                inputStream.close();

                return new Texture(Gdx.files.local("static_map.png"));
            } else {
                Gdx.app.error("GeoapifyMap", "Failed to fetch static map: " + connection.getResponseCode());
            }
        } catch (Exception e) {
            Gdx.app.error("GeoapifyMap", "Error fetching static map", e);
        }
        return null;
    }

    private void handleInput() {
        if (Gdx.input.isKeyJustPressed(Input.Keys.M) || Gdx.input.isKeyJustPressed(Input.Keys.EQUALS)) {
            zoom += 0.5;
            reloadMap();
        } else if (Gdx.input.isKeyJustPressed(Input.Keys.N)) {
            zoom = Math.max(1, zoom - 0.5);
            reloadMap();
        } else if (Gdx.input.isKeyPressed(Input.Keys.LEFT)) {
            centerLon -= 0.01;
            reloadMap();
        } else if (Gdx.input.isKeyPressed(Input.Keys.RIGHT)) {
            centerLon += 0.01;
            reloadMap();
        } else if (Gdx.input.isKeyPressed(Input.Keys.UP)) {
            centerLat += 0.01;
            reloadMap();
        } else if (Gdx.input.isKeyPressed(Input.Keys.DOWN)) {
            centerLat -= 0.01;
            reloadMap();
        }
    }

    private void checkMarkerClick() {
        if (Gdx.input.justTouched()) {
            float touchX = Gdx.input.getX();
            float touchY = Gdx.input.getY();

            Vector2 worldCoords = viewport.unproject(new Vector2(touchX, touchY));

            for (double[] marker : markers) {
                float markerX = (float) ((marker[0] - centerLon) * 800 / zoom);
                float markerY = (float) ((marker[1] - centerLat) * 600 / zoom);

                if (Math.abs(worldCoords.x - markerX) < 60 && Math.abs(worldCoords.y - markerY) < 100) {
                    Gdx.app.log("Marker Clicked", "Switching to MainGameScreen");
                    game.setScreen(mainGameScreen); // Switch to MainGameScreen
                    return;
                }
            }
        }
    }

    private void reloadMap() {
        if (staticMapTexture != null) {
            staticMapTexture.dispose();
        }
        staticMapTexture = fetchStaticMap();
    }

    @Override
    public void show() {
    }

    @Override
    public void render(float delta) {
        handleInput();
        checkMarkerClick();

        if (staticMapTexture == null) {
            return;
        }

        batch.begin();
        batch.draw(staticMapTexture, 0, 0, viewport.getWorldWidth(), viewport.getWorldHeight());
        batch.end();
    }

    @Override
    public void resize(int width, int height) {
        viewport.update(width, height );
    }

    @Override
    public void pause() {
    }

    @Override
    public void resume() {
    }

    @Override
    public void hide() {
    }

    @Override
    public void dispose() {
        if (staticMapTexture != null) {
            staticMapTexture.dispose();
            staticMapTexture = null;
        }
        if (batch != null) {
            batch.dispose();
            batch = null;
        }
    }
}
