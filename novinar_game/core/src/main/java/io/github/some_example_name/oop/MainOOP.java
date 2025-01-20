package io.github.some_example_name.oop;

import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.utils.viewport.FitViewport;
import com.badlogic.gdx.utils.viewport.Viewport;

public class MainOOP extends ApplicationAdapter {
    private OrthographicCamera camera;
    private Viewport viewport;
    private SpriteBatch batch;
    private DynamicTileProvider tileProvider;

    @Override
    public void create() {
        camera = new OrthographicCamera();
        viewport = new FitViewport(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), camera);
        batch = new SpriteBatch();

        // Initialize the tile provider
        tileProvider = new DynamicTileProvider();

        // Set initial camera position
        camera.position.set(0, 0, 0);
        camera.zoom = 1f; // Adjust zoom level as needed
        camera.update();
    }

    @Override
    public void render() {
        handleInput();

        // Clear the screen
        Gdx.gl.glClearColor(0, 0, 0, 1);
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

        // Update the camera
        camera.update();
        batch.setProjectionMatrix(camera.combined);

        // Render the tiles
        batch.begin();
        tileProvider.render(batch,
            camera.position.x - viewport.getWorldWidth() * camera.zoom / 2,
            camera.position.y - viewport.getWorldHeight() * camera.zoom / 2,
            viewport.getWorldWidth() * camera.zoom,
            viewport.getWorldHeight() * camera.zoom);
        batch.end();
    }

    private void handleInput() {
        // Zoom controls
        if (Gdx.input.isKeyPressed(Input.Keys.PLUS)) {
            camera.zoom -= 0.02f;
        }
        if (Gdx.input.isKeyPressed(Input.Keys.MINUS)) {
            camera.zoom += 0.02f;
        }
        camera.zoom = MathUtils.clamp(camera.zoom, 0.5f, 2f);

        // Camera movement
        float moveSpeed = 200 * Gdx.graphics.getDeltaTime();
        if (Gdx.input.isKeyPressed(Input.Keys.LEFT)) {
            camera.translate(-moveSpeed, 0);
        }
        if (Gdx.input.isKeyPressed(Input.Keys.RIGHT)) {
            camera.translate(moveSpeed, 0);
        }
        if (Gdx.input.isKeyPressed(Input.Keys.UP)) {
            camera.translate(0, moveSpeed);
        }
        if (Gdx.input.isKeyPressed(Input.Keys.DOWN)) {
            camera.translate(0, -moveSpeed);
        }
    }

    @Override
    public void dispose() {
        batch.dispose();
        tileProvider.dispose();
    }
}
