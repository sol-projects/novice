package io.github.some_example_name.oop;

import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.Pool;
import com.badlogic.gdx.utils.TimeUtils;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.utils.viewport.FitViewport;
import com.badlogic.gdx.utils.viewport.Viewport;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.GlyphLayout;
import com.badlogic.gdx.math.Rectangle;
import com.badlogic.gdx.graphics.g2d.ParticleEffect;


import java.util.Iterator;

import io.github.some_example_name.assets.AssetDescriptors;
import io.github.some_example_name.assets.GameAssetManager;
import io.github.some_example_name.oop.debug.DebugCameraController;
import io.github.some_example_name.oop.debug.MemoryInfo;
import io.github.some_example_name.oop.debug.ViewportUtils;


public class MainOOP extends ApplicationAdapter {
    private SpriteBatch batch;
    private Car car;
    private Array<RoadSign> roadSigns;
    private Array<Gas> gasList;
    private Array<SpeedBoost> speedBoostList;

    private Background background;
    private BitmapFont font;

    private float timeSurvived;
    private long gameStartTime;

    private long lastGasSpawnTime;
    private long lastSpeedBoostSpawnTime;
    private long lastRoadSignSpawnTime;

    private float gasSpawnInterval = 5f;
    private float speedBoostSpawnInterval = 10f;
    private float roadSignSpawnInterval = 1f;

    private boolean gameOver;
    private GameState currentState;
    private OrthographicCamera camera;

    private DebugCameraController debugCameraController;
    private MemoryInfo memoryInfo;
    private boolean debug = false;

    private ShapeRenderer shapeRenderer;
    public Viewport viewport;
    private ParticleEffect blowBackEffect;
    private  ParticleEffect tire;
    private TextureAtlas atlas;

    private Pool<Gas> gasPool = new Pool<Gas>(5, 10) {
        @Override
        protected Gas newObject() {
            return new Gas(atlas.findRegion("gasoline"));        }
    };

    private Pool<RoadSign> roadSignPool = new Pool<RoadSign>(5, 20) {
        @Override
        protected RoadSign newObject() {
            return new RoadSign(atlas.findRegion("Misc_prop_road_block"));        }
    };

    private Pool<SpeedBoost> speedBoostPool = new Pool<SpeedBoost>(1, 3) {
        @Override
        protected SpeedBoost newObject() {
            return new SpeedBoost(atlas.findRegion("speedBoost"));        }
    };

    public enum GameState {
        PLAYING,
        PAUSED,
        GAME_OVER
    }

    @Override
    public void create() {
        batch = new SpriteBatch();
        GameAssetManager.getInstance().loadAllAssets();

        atlas = GameAssetManager.getInstance().getAssetManager().get(AssetDescriptors.ATLAS);

        blowBackEffect = new ParticleEffect();
        tire = new ParticleEffect();
        blowBackEffect = GameAssetManager.getInstance().getAssetManager().get(AssetDescriptors.PARTICLE_BLOWBACK);
        tire = GameAssetManager.getInstance().getAssetManager().get(AssetDescriptors.PARTICLE_TIRE);        car = new Car(atlas.findRegion("Player_red_single"));
        background = new Background(atlas.findRegion("Desert_road (64 x 64)"));


        blowBackEffect.start();
        tire.start();
        roadSigns = new Array<>();
        speedBoostList = new Array<>();
        gasList = new Array<>();
        font = new BitmapFont();
        gameStartTime = TimeUtils.nanoTime();
        lastGasSpawnTime = gameStartTime;
        lastRoadSignSpawnTime = gameStartTime;
        currentState = GameState.PLAYING;
        gameOver = false;

        camera = new OrthographicCamera();
        camera.setToOrtho(false, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        batch = new SpriteBatch();

        debugCameraController = new DebugCameraController();
        debugCameraController.setStartPosition(Gdx.graphics.getWidth() / 2f, Gdx.graphics.getHeight() / 2f);
        memoryInfo = new MemoryInfo(500);

        shapeRenderer = new ShapeRenderer();
        viewport = new FitViewport(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), camera);

    }

    @Override
    public void render() {
        Gdx.gl.glClearColor(0, 0, 0f, 1);
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

        handleInput();

        if (debug) {
            debugCameraController.handleDebugInput(Gdx.graphics.getDeltaTime());
            memoryInfo.update();
        }

        batch.setProjectionMatrix(camera.combined);
        batch.begin();

        blowBackEffect.setPosition(car.getBounds().x + car.getBounds().width / 2, car.getBounds().y);
        blowBackEffect.flipY();



        background.update(Gdx.graphics.getDeltaTime());
        background.draw(batch);


        if (currentState == GameState.PLAYING) {
            blowBackEffect.update(Gdx.graphics.getDeltaTime());
            blowBackEffect.draw(batch);


            car.update(Gdx.graphics.getDeltaTime());
            batch.draw(car.getTexture(), car.getBounds().x, car.getBounds().y, car.getBounds().width, car.getBounds().height);
            timeSurvived = (TimeUtils.nanoTime() - gameStartTime) / 1_000_000_000f;

            spawnGas();
            spawnRoadSign();
            spawnSpeedBoost();
            handleCollisions();
            updateGas();
            tire.update(Gdx.graphics.getDeltaTime());
            tire.draw(batch);
            drawStatus();
        } else if (currentState == GameState.GAME_OVER) {
            drawGameOver();
        } else if (currentState == GameState.PAUSED) {
            drawPaused();
        }

        batch.end();


        if (debug) {
            debugCameraController.applyTo(camera);
            batch.begin();

                GlyphLayout layout = new GlyphLayout(font, "FPS:" + Gdx.graphics.getFramesPerSecond());
                font.setColor(Color.YELLOW);
                font.draw(batch, layout, Gdx.graphics.getWidth() - layout.width, Gdx.graphics.getHeight() - 50);

                font.setColor(Color.YELLOW);
                font.draw(batch, "RC:" + batch.totalRenderCalls, Gdx.graphics.getWidth() / 2f, Gdx.graphics.getHeight() - 20);

                memoryInfo.render(batch, font);

            batch.end();

            batch.totalRenderCalls = 0;
            ViewportUtils.drawGrid(viewport, shapeRenderer, 50);

            shapeRenderer.setProjectionMatrix(camera.combined);
            shapeRenderer.begin(ShapeRenderer.ShapeType.Line);
            {
                shapeRenderer.setColor(1, 1, 0, 1);

                for (RoadSign roadSign : roadSigns) {
                    Rectangle signBounds = roadSign.getBounds();
                    shapeRenderer.rect(signBounds.x, signBounds.y, signBounds.width, signBounds.height);
                }

                for (Gas gas : gasList) {
                    Rectangle gasBounds = gas.getBounds();
                    shapeRenderer.rect(gasBounds.x, gasBounds.y, gasBounds.width, gasBounds.height);
                }
                for (SpeedBoost speedBoost : speedBoostList) {
                    Rectangle boostBounds = speedBoost.getBounds();
                    shapeRenderer.rect(boostBounds.x, boostBounds.y, boostBounds.width, boostBounds.height);
                }

                Rectangle carBounds = car.getBounds();
                shapeRenderer.rect(carBounds.x, carBounds.y, carBounds.width, carBounds.height);
            }
            shapeRenderer.end();
        }

        //batch.end();
    }

    private void handleInput() {
        if (Gdx.input.isKeyJustPressed(Input.Keys.F1)) debug = !debug;

        if (Gdx.input.isKeyJustPressed(Input.Keys.P)) {
            togglePause();
        }
        if (currentState == GameState.GAME_OVER && Gdx.input.isKeyJustPressed(Input.Keys.R)) {
            resetGame();
        }
    }

    private void togglePause() {
        if (currentState == GameState.PLAYING) {
            currentState = GameState.PAUSED;
        } else if (currentState == GameState.PAUSED) {
            currentState = GameState.PLAYING;
        }
    }

    private void handleCollisions() {
        handleRoadSignCollisions();
        handleSpeedBoostCollisions();
        handleGasCollisions();
    }

    private void handleRoadSignCollisions() {
        for (Iterator<RoadSign> iter = roadSigns.iterator(); iter.hasNext(); ) {
            RoadSign roadSign = iter.next();
            roadSign.update(Gdx.graphics.getDeltaTime());
            batch.draw(roadSign.getTexture(), roadSign.getBounds().x, roadSign.getBounds().y, roadSign.getBounds().width, roadSign.getBounds().height);

            if (roadSign.getBounds().y + roadSign.getBounds().height < 0) {
                iter.remove();
                roadSign.reset();
                roadSignPool.free(roadSign);
            }

            if (roadSign.getBounds().overlaps(car.getBounds())) {
                car.health -= 20;
                iter.remove();
                roadSign.reset();
                roadSignPool.free(roadSign);
            }
        }
    }

    private void handleSpeedBoostCollisions() {
        for (Iterator<SpeedBoost> iter = speedBoostList.iterator(); iter.hasNext(); ) {
            SpeedBoost speedBoost = iter.next();
            speedBoost.update(Gdx.graphics.getDeltaTime());
            batch.draw(speedBoost.getTexture(), speedBoost.getBounds().x, speedBoost.getBounds().y, speedBoost.getBounds().width, speedBoost.getBounds().height);

            if (speedBoost.getBounds().y + speedBoost.getBounds().height < 0) {
                iter.remove();
                speedBoost.reset();
                speedBoostPool.free(speedBoost);
                continue;
            }

            if (speedBoost.getBounds().overlaps(car.getBounds())) {
                car.activateSpeedBoost();
                iter.remove();
                speedBoost.reset();
                speedBoostPool.free(speedBoost);
            }
        }
    }

    private void handleGasCollisions() {
        for (Iterator<Gas> iter = gasList.iterator(); iter.hasNext(); ) {
            Gas gas = iter.next();
            gas.update(Gdx.graphics.getDeltaTime());
            batch.draw(gas.getTexture(), gas.getBounds().x, gas.getBounds().y, gas.getBounds().width, gas.getBounds().height);
            tire.setPosition(gas.getBounds().x, gas.getBounds().y);

            if (gas.getBounds().y + gas.getBounds().height < 0) {
                iter.remove();
                gas.reset();
                gasPool.free(gas);
                //tire.dispose();
            }

            if (gas.getBounds().overlaps(car.getBounds())) {
                car.gasRemaining += 20;
                iter.remove();
                gas.reset();
                gasPool.free(gas);
                tire.dispose();
            }
        }
    }

    private void updateGas() {
        if (currentState == GameState.PLAYING) {
            car.gasRemaining -= 5 * Gdx.graphics.getDeltaTime();
            if (car.gasRemaining <= 0) {
                car.gasRemaining = 0;
                currentState = GameState.GAME_OVER;
            }
            if (car.health <= 0) {
                currentState = GameState.GAME_OVER;
            }
        }
    }

    private void drawStatus() {
        font.draw(batch, "GAS: " + (int) car.gasRemaining, 20, Gdx.graphics.getHeight() - 20);
        font.draw(batch, "HEALTH: " + car.health, 20, Gdx.graphics.getHeight() - 60);
        font.draw(batch, "TIME SURVIVED: " + (int) timeSurvived + " SECONDS", 20, Gdx.graphics.getHeight() - 100);
    }

    private void drawGameOver() {
        font.draw(batch, "GAME OVER", Gdx.graphics.getWidth() / 2 - 50, Gdx.graphics.getHeight() / 2);
        font.draw(batch, "TIME SURVIVED: " + (int) timeSurvived + " SECONDS", Gdx.graphics.getWidth() / 2 - 100, Gdx.graphics.getHeight() / 2 - 40);
        font.draw(batch, "PRESS R TO RESET", Gdx.graphics.getWidth() / 2 - 100, Gdx.graphics.getHeight() / 2 - 80);
    }

    private void drawPaused() {
        font.draw(batch, "PAUSED", Gdx.graphics.getWidth() / 2 - 30, Gdx.graphics.getHeight() / 2);
    }

    private boolean isOverlapping(GameObject newObject) {
        for (Gas gas : gasList) {
            if (newObject.getBounds().overlaps(gas.getBounds())) {
                return true;
            }
        }
        for (RoadSign roadSign : roadSigns) {
            if (newObject.getBounds().overlaps(roadSign.getBounds())) {
                return true;
            }
        }
        for (SpeedBoost speedBoost : speedBoostList) {
            if (newObject.getBounds().overlaps(speedBoost.getBounds())) {
                return true;
            }
        }
        return false;
    }

    private void spawnGas() {
        if (TimeUtils.nanoTime() - lastGasSpawnTime > gasSpawnInterval * 1_000_000_000L) {
            Gas gas = gasPool.obtain();
            do {
                gas.setPosition(MathUtils.random(0, Gdx.graphics.getWidth() - gas.getBounds().width), Gdx.graphics.getHeight());
            } while (isOverlapping(gas));

            gasList.add(gas);
            lastGasSpawnTime = TimeUtils.nanoTime();
            gasSpawnInterval = MathUtils.random(8f, 12f);
        }
    }

    private void spawnRoadSign() {
        if (TimeUtils.nanoTime() - lastRoadSignSpawnTime > roadSignSpawnInterval * 1_000_000_000L) {
            Gdx.app.log("Spawn", "Spawning new RoadSign");
            RoadSign roadSign = roadSignPool.obtain();
            do {
                roadSign.setPosition(MathUtils.random(0, Gdx.graphics.getWidth() - roadSign.getBounds().width), Gdx.graphics.getHeight());
            } while (isOverlapping(roadSign));

            roadSigns.add(roadSign);
            lastRoadSignSpawnTime = TimeUtils.nanoTime();
        }
    }

    private void spawnSpeedBoost() {
        if (TimeUtils.nanoTime() - lastSpeedBoostSpawnTime > speedBoostSpawnInterval * 1_000_000_000L) {
            SpeedBoost speedBoost = speedBoostPool.obtain();
            do {
                speedBoost.setPosition(MathUtils.random(0, Gdx.graphics.getWidth() - speedBoost.getBounds().width), Gdx.graphics.getHeight());
            } while (isOverlapping(speedBoost));

            speedBoostList.add(speedBoost);
            lastSpeedBoostSpawnTime = TimeUtils.nanoTime();
            speedBoostSpawnInterval = MathUtils.random(15f, 25f);
        }
    }

    private void resetGame() {
        car.reset();
        currentState = GameState.PLAYING;
        timeSurvived = 0;
        gameStartTime = TimeUtils.nanoTime();
        lastGasSpawnTime = gameStartTime;
        lastRoadSignSpawnTime = gameStartTime;
        roadSigns.clear();
        gasList.clear();
        speedBoostList.clear();
        gameOver = false;
    }

    @Override
    public void dispose() {
        batch.dispose();
        font.dispose();
        GameAssetManager.getInstance().dispose();

    }
}
