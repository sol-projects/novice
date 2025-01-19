package io.github.some_example_name.oop;

import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.maps.tiled.renderers.OrthogonalTiledMapRenderer;

public class MapRenderer {
    private OrthogonalTiledMapRenderer mapRenderer;

    public MapRenderer(OrthogonalTiledMapRenderer mapRenderer) {
        this.mapRenderer = mapRenderer;
    }

    public void render(OrthographicCamera camera, SpriteBatch batch) {
        // Set the camera and render the map
        mapRenderer.setView(camera);
        mapRenderer.render();
    }
}

