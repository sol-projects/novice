package io.github.some_example_name.oop;

import com.badlogic.gdx.Game;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.utils.viewport.FitViewport;
import com.badlogic.gdx.utils.viewport.Viewport;

public class MainOOP extends Game {
    private SpriteBatch batch;
    private OrthographicCamera camera;
    private Viewport viewport;

    @Override
    public void create() {
        batch = new SpriteBatch();
        camera = new OrthographicCamera();
        viewport = new FitViewport(800, 480, camera);
        MainGameScreen mainGameScreen = new MainGameScreen(batch, camera, viewport);
        GeoapifyMapScreen geoapifyMapScreen = new GeoapifyMapScreen(this, mainGameScreen);
        setScreen(new MenuScreen(this, mainGameScreen, geoapifyMapScreen));
    }

    @Override
    public void render() {
        super.render();
    }

    @Override
    public void resize(int width, int height) {
        super.resize(width, height);
        viewport.update(width, height);
    }

    @Override
    public void dispose() {
        super.dispose();
        batch.dispose();
    }
}
