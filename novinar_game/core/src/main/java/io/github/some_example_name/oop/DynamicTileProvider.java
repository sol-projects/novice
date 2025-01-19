package io.github.some_example_name.oop;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.maps.tiled.tiles.StaticTiledMapTile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

public class DynamicTileProvider {
    private static final int TILE_SIZE = 256; // Tile size in pixels
    private static final int ZOOM = 3; // Fixed zoom level
    private final String geoapifyBaseUrl = "https://maps.geoapify.com/v1/tile/carto/{z}/{x}/{y}.png?&apiKey=283c4e292efd4ec89d370fc61f2ecb05";
    private final Map<String, StaticTiledMapTile> tileCache = new HashMap<>();

    public void render(SpriteBatch batch, float cameraX, float cameraY, float viewportWidth, float viewportHeight) {
        // Calculate the range of tiles to render
        int minTileX = (int) Math.floor(cameraX / TILE_SIZE);
        int minTileY = (int) Math.floor(cameraY / TILE_SIZE);
        int maxTileX = (int) Math.ceil((cameraX + viewportWidth) / TILE_SIZE);
        int maxTileY = (int) Math.ceil((cameraY + viewportHeight) / TILE_SIZE);

        // Clamp the tile indices to valid ranges
        int maxIndex = (1 << ZOOM) - 1; // Maximum index for the given zoom level
        minTileX = Math.max(0, Math.min(minTileX, maxIndex));
        minTileY = Math.max(0, Math.min(minTileY, maxIndex));
        maxTileX = Math.max(0, Math.min(maxTileX, maxIndex));
        maxTileY = Math.max(0, Math.min(maxTileY, maxIndex));

        // Debugging: Log the tile range
        System.out.println("Rendering tiles: [" + minTileX + "," + minTileY + "] to [" + maxTileX + "," + maxTileY + "]");

        // Render all tiles in the range
        for (int x = minTileX; x <= maxTileX; x++) {
            for (int y = minTileY; y <= maxTileY; y++) {
                StaticTiledMapTile tile = getTile(x, y, ZOOM);
                if (tile != null) {
                    // Debugging: Log tile rendering
                    System.out.println("Drawing tile at: " + (x * TILE_SIZE) + ", " + (y * TILE_SIZE));
                    batch.draw(tile.getTextureRegion(), x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
                }
            }
        }
    }

    public StaticTiledMapTile getTile(int x, int y, int zoom) {
        String cacheKey = zoom + "_" + x + "_" + y;

        // Check if tile is already cached
        if (tileCache.containsKey(cacheKey)) {
            return tileCache.get(cacheKey);
        }

        HttpURLConnection connection = null;
        InputStream inputStream = null;
        try {
            // Prepare the URL for the tile
            String tileUrl = geoapifyBaseUrl.replace("{z}", String.valueOf(zoom))
                .replace("{x}", String.valueOf(x))
                .replace("{y}", String.valueOf(y));
            System.out.println("Loading tile: " + tileUrl);

            connection = (HttpURLConnection) new URL(tileUrl).openConnection();
            connection.setRequestMethod("GET");
            connection.connect();

            // Check if the response is successful
            if (connection.getResponseCode() == HttpURLConnection.HTTP_OK) {
                inputStream = connection.getInputStream();

                // Create a temporary file from InputStream
                File tempFile = File.createTempFile("tile", ".png");
                FileOutputStream outputStream = new FileOutputStream(tempFile);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                outputStream.close();

                // Load texture from the temporary file
                Texture texture = new Texture(tempFile.getAbsolutePath());
                tempFile.deleteOnExit();

                // Convert Texture to TextureRegion and return StaticTiledMapTile
                TextureRegion region = new TextureRegion(texture);
                StaticTiledMapTile tile = new StaticTiledMapTile(region);

                // Cache the tile
                tileCache.put(cacheKey, tile);

                System.out.println("Tile successfully loaded: [" + x + ", " + y + "]");

                return tile;
            } else {
                System.err.println("Failed to load tile: " + connection.getResponseCode());
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (inputStream != null) inputStream.close();
                if (connection != null) connection.disconnect();
            } catch (Exception ignored) {}
        }

        return null; // Return null if loading failed
    }

    public void dispose() {
        // Dispose of all cached tiles
        for (StaticTiledMapTile tile : tileCache.values()) {
            TextureRegion region = tile.getTextureRegion();
            if (region != null && region.getTexture() != null) {
                region.getTexture().dispose();
            }
        }
        tileCache.clear();
    }
}
