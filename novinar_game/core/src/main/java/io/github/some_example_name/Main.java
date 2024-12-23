package io.github.some_example_name;

import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.audio.Sound;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.math.Rectangle;
import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.ScreenUtils;
import com.badlogic.gdx.utils.TimeUtils;
import com.badlogic.gdx.utils.viewport.FitViewport;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.utils.viewport.Viewport;
import io.github.some_example_name.oop.debug.ViewportUtils;

import java.util.Iterator;

/**
 * {@link com.badlogic.gdx.ApplicationListener} implementation shared by all platforms.
 */
public class Main extends ApplicationAdapter {
    private static final float WORLD_WIDTH = 80;
    private static final float WORLD_HEIGHT = 60;
    private SpriteBatch batch;
    private Texture carImg;
    private Texture roadSingImg;
    private Texture gasImg;



    private Texture backgroundImg;
    private Texture backgroundImgGas;

    private Rectangle car;
    private Array<Rectangle> gass;
    private Array<Rectangle> roadSings;


    private float gasSpawnTime;
    private double gasRemaining;
    private long gameStartTime;
    private float timeSurvived;

    private float roadSingSpawnTime;
    private int health;

    private Sound gasSound;
    private BitmapFont font;

    private float road1Y;
    private float road2Y;

    private static final float CAR_SPEED = 250f;

    private static final float GAS_SPEED = 100f;
    private static final float GAS_SPAWN_TIME = 10f;
    private static final float ROAD_SING_SPEED = 100f;
    private static final float ROAD_SING_DAMAGE = 25f;
    private static final float ROAD_SING_SPAWN_TIME = 2f;
    private static final double GAS_DECREASE_RATE = 0.1;
    private static final float SCROLL_SPEED = 100f;
    private OrthographicCamera camera;
    private Viewport viewport;
    private Viewport hudViewport;
//    private ShapeRenderer renderer;


    @Override
    public void create() {
        batch = new SpriteBatch();

        carImg = new Texture("Mini Pixel Pack 2/Cars/Player_red_single.png");
        roadSingImg = new Texture("Mini Pixel Pack 2/Props/Misc_prop_road_block.png");
        gasImg = new Texture("gasoline.png");
        backgroundImg = new Texture("Mini Pixel Pack 2/Levels/Desert_road (64 x 64).png");
        backgroundImgGas = new Texture("Mini Pixel Pack 2/Levels/Winter_road (64 x 64).png");


        camera = new OrthographicCamera();
        viewport = new FitViewport(WORLD_WIDTH, WORLD_HEIGHT, camera);
        hudViewport = new FitViewport(Gdx.graphics.getWidth(), Gdx.graphics.getHeight());


        car = new Rectangle();
        car.x = hudViewport.getWorldWidth() / 2f - carImg.getWidth() / 2f;
        car.y = 20f;
        car.width = carImg.getWidth()*4f;
        car.height = carImg.getHeight()*4f;

        gass = new Array<>();
        gasRemaining = 100;
        spawnGas();

        roadSings = new Array<>();
        health = 100;
        spawnRoadSing();


        //gasSound = Gdx.audio.newSound(Gdx.files.internal("gas_pickup.wav"));
        font = new BitmapFont();

        gameStartTime = TimeUtils.nanoTime();
    }

    @Override
    public void resize(int width, int height) {
        viewport.update(width, height, true);
        hudViewport.update(width, height, true);
        ViewportUtils.debugPixelsPerUnit(viewport);
    }


    @Override
    public void render() {
        Gdx.gl.glClearColor(0, 0, 0.5f, 1);
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

        if (gasRemaining > 0 && health > 0) {
            handleInput();
            update(Gdx.graphics.getDeltaTime());
            timeSurvived = (TimeUtils.nanoTime() - gameStartTime) / 1_000_000_000f;
        }

        hudViewport.apply();
        batch.setProjectionMatrix(hudViewport.getCamera().combined);

        batch.begin();
        draw();
        batch.end();
    }

    private void handleInput() {
        if (Gdx.input.isKeyPressed(Input.Keys.LEFT)) moveLeft(Gdx.graphics.getDeltaTime());
        if (Gdx.input.isKeyPressed(Input.Keys.RIGHT)) moveRight(Gdx.graphics.getDeltaTime());
    }

    private void update(float delta) {
        float elapsedTime = TimeUtils.nanoTime() / 1_000_000_000f;

        if (elapsedTime - gasSpawnTime > GAS_SPAWN_TIME) spawnGas();
        if (elapsedTime - roadSingSpawnTime > ROAD_SING_SPAWN_TIME) spawnRoadSing();

        // Decrease gas over time
        gasRemaining -= GAS_DECREASE_RATE;
        if (gasRemaining < 0) gasRemaining = 0;


        for (Iterator<Rectangle> it = gass.iterator(); it.hasNext(); ) {
            Rectangle gas = it.next();
            gas.y -= GAS_SPEED * delta;
            if (gas.y + gasImg.getHeight() < 0) {
                it.remove();
            }
            if (gas.overlaps(car)) {
                gasRemaining += 40;
                //gasSound.play();
                it.remove();
            }
        }

        for (Iterator<Rectangle> it = roadSings.iterator(); it.hasNext(); ) {
            Rectangle roadSing = it.next();
            roadSing.y -= ROAD_SING_SPEED * delta;
            if (roadSing.y + roadSingImg.getHeight() < 0) {
                it.remove();
            }
            if (roadSing.overlaps(car)) {
                health -= (int) ROAD_SING_DAMAGE;
                it.remove();
            }
        }
    }

    private void draw() {

        int screenWidth = Gdx.graphics.getWidth();
        int screenHeight = Gdx.graphics.getHeight();
        float worldWidth = viewport.getWorldWidth();
        float worldHeight = viewport.getWorldHeight();

        String screenSize = "Screen/Window size: " + screenWidth + " x " + screenHeight + " px";
        String worldSize = "World size: " + (int) worldWidth + " x " + (int) worldHeight + " world units";
        String oneWorldUnit = "One world unit: " + (screenWidth / worldWidth) + " x " + (screenHeight / worldHeight) + " px";


        font.draw(batch,
            screenSize,
            20f,
            hudViewport.getWorldHeight() - 120f);

        font.draw(batch,
            worldSize,
            20f,
            hudViewport.getWorldHeight() - 150f);

        font.draw(batch,
            oneWorldUnit,
            20f,
            hudViewport.getWorldHeight() - 180f);

        if (gasRemaining <= 0 || health <= 0) {
            font.setColor(Color.RED);
            font.draw(batch, "GAME OVER", hudViewport.getWorldWidth() / 2f - 50, hudViewport.getWorldHeight() / 2f);
            font.draw(batch, "TIME SURVIVED: " + (int) timeSurvived + " SECONDS", hudViewport.getWorldWidth()/ 2f - 100, hudViewport.getWorldHeight()/ 2f - 40);
            return;
        }

        for (Rectangle gas : gass) {
            batch.draw(gasImg, gas.x, gas.y, gas.width, gas.height);
        }

        for (Rectangle roadSing : roadSings) {
            batch.draw(roadSingImg, roadSing.x, roadSing.y, roadSingImg.getWidth()*5f, roadSingImg.getHeight()*5f);
        }

        batch.draw(carImg, car.x, car.y, carImg.getWidth()*4f, carImg.getHeight()*4f);

        font.setColor(Color.RED);
        font.draw(batch, "HEALTH: " + health, 20f, hudViewport.getWorldHeight() - 20f);

        font.setColor(Color.YELLOW);
        font.draw(batch, "GAS REMAINING: " + (int) gasRemaining, 20f, hudViewport.getWorldHeight()- 60f);

        font.setColor(Color.GREEN);
        font.draw(batch, "TIME SURVIVED: " + (int) timeSurvived + " SECONDS", 20f, hudViewport.getWorldHeight() - 100f);
    }

    private void moveLeft(float delta) {
        car.x -= CAR_SPEED * delta;
        if (car.x < 0)
            car.x = 0f;
    }

    private void moveRight(float delta) {
        car.x += CAR_SPEED * delta;
        if (car.x > hudViewport.getWorldWidth() - carImg.getWidth())
            car.x = hudViewport.getWorldWidth() - carImg.getWidth();
    }

    private void spawnGas() {
        Rectangle gas = new Rectangle();
        gas.x = MathUtils.random(0f, hudViewport.getWorldWidth() - gasImg.getWidth());
        gas.y = hudViewport.getWorldHeight();
        gas.width = gasImg.getWidth()/4f;
        gas.height = gasImg.getHeight()/4f;
        gass.add(gas);
        gasSpawnTime = TimeUtils.nanoTime() / 1_000_000_000f;
    }

    private void spawnRoadSing() {
        Rectangle roadSing = new Rectangle();
        roadSing.x = MathUtils.random(0f, hudViewport.getWorldWidth() - roadSingImg.getWidth());
        roadSing.y = hudViewport.getWorldHeight();
        roadSing.width = roadSingImg.getWidth()*5f;
        roadSing.height = roadSingImg.getHeight()*5f;
        roadSings.add(roadSing);
        roadSingSpawnTime = TimeUtils.nanoTime() / 1_000_000_000f;
    }

    @Override
    public void dispose() {
        batch.dispose();
        carImg.dispose();
        gasImg.dispose();
        roadSingImg.dispose();
        backgroundImg.dispose();
        gasSound.dispose();
        font.dispose();
    }
}
