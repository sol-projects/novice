package io.github.some_example_name.oop;

import com.badlogic.gdx.maps.tiled.TiledMap;
import com.badlogic.gdx.maps.tiled.TmxMapLoader;
import com.badlogic.gdx.utils.Disposable;

public class MapLoader implements Disposable {
    private TiledMap currentMap;

    /**
     * Loads a map for the given zoom level.
     *
     * @param zoomLevel The zoom level of the map to load.
     * @return The loaded TiledMap.
     */
    public TiledMap loadMap(int zoomLevel) {
        // Dispose of the currently loaded map if one exists
        if (currentMap != null) {
            currentMap.dispose();
        }

        // Construct the map path based on zoom level
        String mapPath = "tiled/Map.tmx";

        try {
            // Load the map
            currentMap = new TmxMapLoader().load(mapPath);
        } catch (Exception e) {
            // Handle the case where the map file does not exist or fails to load
            System.err.println("Failed to load map: " + mapPath);
            e.printStackTrace();
            currentMap = null;
        }

        return currentMap;
    }

    /**
     * Disposes of the currently loaded map and cleans up resources.
     */
    @Override
    public void dispose() {
        if (currentMap != null) {
            currentMap.dispose();
            currentMap = null;
        }
    }
}
