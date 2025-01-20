package io.github.some_example_name.oop;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;

import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

public class DynamicTileProvider {
    private static final int TILE_SIZE = 256; // Standard tile size
    private static final int ZOOM = 3; // Fixed zoom level for simplicity
    private final String baseUrl = "https://maps.geoapify.com/v1/tile/carto/{z}/{x}/{y}.png?&apiKey=283c4e292efd4ec89d370fc61f2ecb05";
    private final Map<String, Texture> tileCache = new HashMap<>();

    public void render(SpriteBatch batch, float xStart, float yStart, float width, float height) {
        int minTileX = (int) Math.floor(xStart / TILE_SIZE);
        int minTileY = (int) Math.floor(yStart / TILE_SIZE);
        int maxTileX = (int) Math.ceil((xStart + width) / TILE_SIZE);
        int maxTileY = (int) Math.ceil((yStart + height) / TILE_SIZE);

        for (int x = minTileX; x <= maxTileX; x++) {
            for (int y = minTileY; y <= maxTileY; y++) {
                String key = ZOOM + "_" + x + "_" + y;
                int finalX = x;
                int finalY = y;
                Texture texture = tileCache.computeIfAbsent(key, k -> fetchTile(finalX, finalY, ZOOM));

                if (texture != null) {
                    batch.draw(new TextureRegion(texture), x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
                }
            }
        }
    }

    private Texture fetchTile(int x, int y, int zoom) {
        try {
            String url = baseUrl.replace("{z}", String.valueOf(zoom))
                .replace("{x}", String.valueOf(x))
                .replace("{y}", String.valueOf(y));

            HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
            connection.setRequestMethod("GET");

            if (connection.getResponseCode() == HttpURLConnection.HTTP_OK) {
                return new Texture(String.valueOf(new URL(url).openStream()));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public void dispose() {
        for (Texture texture : tileCache.values()) {
            texture.dispose();
        }
    }
}
