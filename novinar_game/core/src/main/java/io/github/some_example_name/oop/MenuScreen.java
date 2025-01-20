package io.github.some_example_name.oop;

import com.badlogic.gdx.Game;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.math.Rectangle;

public class MenuScreen implements Screen {

    private final Game game;
    private final MainGameScreen mainGameScreen;
    private final GeoapifyMapScreen mapScreen;
    private SpriteBatch batch;
    private ShapeRenderer shapeRenderer;
    private BitmapFont font;

    private Rectangle playButtonBounds;
    private Rectangle mapButtonBounds;

    public MenuScreen(Game game, MainGameScreen mainGameScreen, GeoapifyMapScreen mapScreen) {
        this.game = game;
        this.mainGameScreen = mainGameScreen;
        this.mapScreen = mapScreen;
        this.batch = new SpriteBatch();
        this.shapeRenderer = new ShapeRenderer();
        this.font = new BitmapFont();

        float buttonWidth = 200;
        float buttonHeight = 60;
        float centerX = Gdx.graphics.getWidth() / 2f;
        float centerY = Gdx.graphics.getHeight() / 2f;

        this.playButtonBounds = new Rectangle(centerX - buttonWidth / 2, centerY + 40, buttonWidth, buttonHeight);
        this.mapButtonBounds = new Rectangle(centerX - buttonWidth / 2, centerY - 80, buttonWidth, buttonHeight);
    }

    @Override
    public void show() {
    }

    @Override
    public void render(float delta) {
        Gdx.gl.glClearColor(0.1f, 0.1f, 0.3f, 1);
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

        handleInput();

        shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
        // Draw play button
        shapeRenderer.setColor(Color.DARK_GRAY);
        shapeRenderer.rect(playButtonBounds.x, playButtonBounds.y, playButtonBounds.width, playButtonBounds.height);
        // Draw map button
        shapeRenderer.setColor(Color.DARK_GRAY);
        shapeRenderer.rect(mapButtonBounds.x, mapButtonBounds.y, mapButtonBounds.width, mapButtonBounds.height);
        shapeRenderer.end();

        shapeRenderer.begin(ShapeRenderer.ShapeType.Line);
        shapeRenderer.setColor(Color.WHITE);
        shapeRenderer.rect(playButtonBounds.x, playButtonBounds.y, playButtonBounds.width, playButtonBounds.height);
        shapeRenderer.rect(mapButtonBounds.x, mapButtonBounds.y, mapButtonBounds.width, mapButtonBounds.height);
        shapeRenderer.end();

        batch.begin();
        font.setColor(Color.WHITE);
        font.getData().setScale(1.5f);
        // Draw text for play button
        font.draw(batch, "Start Game", playButtonBounds.x + 50, playButtonBounds.y + 40);
        // Draw text for map button
        font.draw(batch, "View Map", mapButtonBounds.x + 50, mapButtonBounds.y + 40);
        batch.end();
    }

    private void handleInput() {
        if (Gdx.input.justTouched()) {
            float x = Gdx.input.getX();
            float y = Gdx.graphics.getHeight() - Gdx.input.getY(); // Flip Y-axis

            if (playButtonBounds.contains(x, y)) {
                game.setScreen(mainGameScreen);
            } else if (mapButtonBounds.contains(x, y)) {
                game.setScreen(mapScreen);
            }
        }
    }

    @Override
    public void resize(int width, int height) {
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
        batch.dispose();
        shapeRenderer.dispose();
        font.dispose();
    }
}
