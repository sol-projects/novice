package io.github.some_example_name.oop;

import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.utils.viewport.FitViewport;
import com.badlogic.gdx.utils.viewport.Viewport;

public class MainOOP extends ApplicationAdapter {
    private OrthographicCamera camera;
    private Viewport viewport;
    private SpriteBatch batch;
    private DynamicTileProvider tileProvider;

    @Override
    public void create() {
        // Initialize camera and viewport
        camera = new OrthographicCamera();
        viewport = new FitViewport(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), camera);
        batch = new SpriteBatch();
        tileProvider = new DynamicTileProvider();

        // Center the camera on the map
        float mapSizeInPixels = (1 << 3) * 256; // Map size for zoom level 3
        camera.position.set(mapSizeInPixels / 2, mapSizeInPixels / 2, 0);
    }

    @Override
    public void render() {
        // Update the camera
        camera.update();

        // Clear the screen
        Gdx.gl.glClearColor(0, 0, 0, 1);
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

        // Render tiles
        batch.setProjectionMatrix(camera.combined);
        batch.begin();
        tileProvider.render(batch, camera.position.x - viewport.getWorldWidth() / 2, camera.position.y - viewport.getWorldHeight() / 2, viewport.getWorldWidth(), viewport.getWorldHeight());
        batch.end();
    }

    @Override
    public void dispose() {
        batch.dispose();
        tileProvider.dispose();
    }
}
