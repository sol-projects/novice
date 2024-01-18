package com.mygdx.game;

import static com.mygdx.game.ApiConnectionKt.sendGet;

import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.InputMultiplexer;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.Mesh;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.graphics.glutils.ShaderProgram;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.input.GestureDetector;
import com.badlogic.gdx.maps.MapLayers;
import com.badlogic.gdx.maps.tiled.TiledMap;
import com.badlogic.gdx.maps.tiled.TiledMapRenderer;
import com.badlogic.gdx.maps.tiled.TiledMapTileLayer;
import com.badlogic.gdx.maps.tiled.renderers.OrthogonalTiledMapRenderer;
import com.badlogic.gdx.maps.tiled.tiles.StaticTiledMapTile;
import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.actions.Actions;
import com.badlogic.gdx.scenes.scene2d.actions.SequenceAction;
import com.badlogic.gdx.scenes.scene2d.ui.Image;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.ui.TextButton;
import com.badlogic.gdx.scenes.scene2d.utils.ClickListener;
import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.ScreenUtils;
import com.badlogic.gdx.utils.viewport.FitViewport;
import com.mygdx.game.lang.Context;
import com.mygdx.game.lang.GeolangKt;
import com.mygdx.game.lang.Renderer;
import com.mygdx.game.utils.Constants;
import com.mygdx.game.utils.Geolocation;
import com.mygdx.game.utils.MapRasterTiles;
import com.mygdx.game.utils.ZoomXY;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ProjectTest extends ApplicationAdapter implements GestureDetector.GestureListener {//Gesturelistener Enables to handle touch gestures.

    private ShapeRenderer shapeRenderer;
    private Vector3 touchPosition;

    private TiledMap tiledMap;
    private TiledMapRenderer tiledMapRenderer;
    private OrthographicCamera camera;

    private Texture[] mapTiles;
    private ZoomXY beginTile;   // top left tile

    private SpriteBatch spriteBatch;

    // buttons
    private FitViewport hudViewport;
    private Stage hudStage;
    private Skin skin;
    private boolean showLangExample = false;

    // animation
    private Stage stage;
    private FitViewport viewport;

    // boat animation
    Geolocation[] boatCoordinates = {
            new Geolocation(46.5602f, 15.625186f),
            new Geolocation(46.5580f, 15.632482f),
            new Geolocation(46.5560f, 15.639112f),
            new Geolocation(46.5555f, 15.647974f),
            new Geolocation(46.5553f, 15.657566f)
    };
    BoatAnimation boatAnimation;

    // center geolocation
    private final Geolocation CENTER_GEOLOCATION = new Geolocation(46.119944,14.815333);

    private ArrayList<INews> news;
    private String code;
    private String current_geojson_code;

    private Mesh geojson_points_mesh;
    private ShaderProgram geojson_points_shader;

    private ArrayList<Geolocation> code_points;
    @Override
    public void create() {
        news = sendGet();
        Iterator<INews> iterator = news.iterator();

        while (iterator.hasNext()) {
            INews currentNews = iterator.next();
            double firstCoordinate = currentNews.getLocation().getCoordinates().getFirst();
            double secondCoordinate = currentNews.getLocation().getCoordinates().getSecond();

            if (firstCoordinate == 0.0 || secondCoordinate == 0.0) {
                iterator.remove();
            }
        }

        code = GeolangKt.load();
        System.out.println(code);

        String geojson = GeolangKt.get_geojson_from_interpreter(code);
        System.out.println(geojson);

        code_points = GeolangKt.get_points_from_geojson(geojson);

        shapeRenderer = new ShapeRenderer();

        camera = new OrthographicCamera();
        camera.setToOrtho(false, Constants.MAP_WIDTH, Constants.MAP_HEIGHT);
        camera.position.set(Constants.MAP_WIDTH / 2f, Constants.MAP_HEIGHT / 2f, 0);
        camera.viewportWidth = Constants.MAP_WIDTH / 2f;
        camera.viewportHeight = Constants.MAP_HEIGHT / 2f;
        camera.zoom = 2f;
        camera.update();

        spriteBatch = new SpriteBatch();
        hudViewport = new FitViewport(Constants.HUD_WIDTH, Constants.HUD_HEIGHT);
        viewport = new FitViewport(Constants.MAP_WIDTH / 2f, Constants.MAP_HEIGHT / 2f, camera);

        touchPosition = new Vector3();

        try {//TU SE ZLADA MAPA
            //in most cases, geolocation won't be in the center of the tile because tile borders are predetermined (geolocation can be at the corner of a tile)
            ZoomXY centerTile = MapRasterTiles.getTileNumber(CENTER_GEOLOCATION.lat, CENTER_GEOLOCATION.lng, Constants.ZOOM);
            mapTiles = MapRasterTiles.getRasterTileZone(centerTile, Constants.NUM_TILES);
            //you need the beginning tile (tile on the top left corner) to convert geolocation to a location in pixels.
            //beginingTILE=the code calculates the beginning tile (the top-left tile)
            beginTile = new ZoomXY(Constants.ZOOM, centerTile.x - ((Constants.NUM_TILES - 1) / 2), centerTile.y - ((Constants.NUM_TILES - 1) / 2));
        } catch (IOException e) {
            e.printStackTrace();
        }
        //TU SE USTVARI MAPA
        tiledMap = new TiledMap();
        MapLayers layers = tiledMap.getLayers();

        TiledMapTileLayer layer = new TiledMapTileLayer(Constants.NUM_TILES, Constants.NUM_TILES, MapRasterTiles.TILE_SIZE, MapRasterTiles.TILE_SIZE);
        int index = 0;
        for (int j = Constants.NUM_TILES - 1; j >= 0; j--) {
            for (int i = 0; i < Constants.NUM_TILES; i++) {
                TiledMapTileLayer.Cell cell = new TiledMapTileLayer.Cell();
                cell.setTile(new StaticTiledMapTile(new TextureRegion(mapTiles[index], MapRasterTiles.TILE_SIZE, MapRasterTiles.TILE_SIZE)));
                layer.setCell(i, j, cell);
                index++;
            }
        }
        layers.add(layer);

        tiledMapRenderer = new OrthogonalTiledMapRenderer(tiledMap);

        // buttons
        skin = new Skin(Gdx.files.internal("ui/uiskin.json"));
        hudStage = new Stage(hudViewport, spriteBatch);
        hudStage.addActor(createButtons());

        Gdx.input.setInputProcessor(new InputMultiplexer(hudStage, new GestureDetector(this)));

        // boat
        boatAnimation = new BoatAnimation(boatCoordinates, beginTile, 5);
        stage = new Stage(viewport, spriteBatch);
        stage.addActor(boatAnimation.create());
        geojson_points_mesh = GeolangKt.create_point_mesh(shapeRenderer, code_points, beginTile);

        String vertexShader = "attribute vec4 a_position;\n" +
                "uniform mat4 u_projTrans;\n" +
                "void main() {\n" +
                "    gl_Position = u_projTrans * a_position;\n" +
                "}";

        String fragmentShader = "#ifdef GL_ES\n" +
                "precision mediump float;\n" +
                "#endif\n" +
                "void main() {\n" +
                "gl_FragColor = vec4(40.0/255.0, 67.0/255.0, 135.0/255.0, 1.0);\n" +
                "}";

        geojson_points_shader = new ShaderProgram(vertexShader, fragmentShader);
        if (!geojson_points_shader.isCompiled()) {
            Gdx.app.error("Shader", geojson_points_shader.getLog());
            Gdx.app.exit();
        }
        Gdx.gl.glEnable(GL20.GL_BLEND);
        Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);
    }

    @Override
    public void render() {
        ScreenUtils.clear(0, 0, 0, 1);

        handleInput();

        camera.update();

        tiledMapRenderer.setView(camera);
        tiledMapRenderer.render();

        hudStage.act(Gdx.graphics.getDeltaTime());
        stage.act(Gdx.graphics.getDeltaTime());

        hudStage.draw();
        stage.draw();

        drawMarkers();

        // lang
        if(showLangExample){
            Renderer renderer = new Renderer();
            try {
                renderer.render(new FileInputStream(new File("program.txt")), new Context(shapeRenderer, camera, beginTile));
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private void drawMarkers() {
        // DA NE POZABIM
        // IDEJA:
        // 1. "markerji" se izrisujejo kot kategorije npr. novice/web-app/src/assets/
        // 2. animacija: dež ki pada ob kategoriji "toča/dež..."
        //    pomoje zelo simpl, 3 kaplje narišeš pa sam uporabiš moveTo dol po y za ene 20 pisklov
        //    potem se pa to loopa
        shapeRenderer.setProjectionMatrix(camera.combined);
        shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
        for (int i = 0; i < news.size(); i++) {
            Vector2 marker = MapRasterTiles.getPixelPosition(
                    news.get(i).getLocation().getCoordinates().getSecond(),
                    news.get(i).getLocation().getCoordinates().getFirst(),
                    beginTile.x, beginTile.y);

            shapeRenderer.setColor(Color.RED);
            shapeRenderer.circle(marker.x, marker.y, 20);
        }
        shapeRenderer.end();

        /*geojson_points_shader.bind();
        geojson_points_shader.setUniformMatrix("u_projTrans", camera.combined);
        geojson_points_mesh.render(geojson_points_shader, GL20.GL_POINTS);*/

        shapeRenderer.begin(ShapeRenderer.ShapeType.Line);
        shapeRenderer.setColor(Color.BLUE);
        for (int i = 0; i < code_points.size(); i++) {
            Vector2 marker = MapRasterTiles.getPixelPosition(
                    code_points.get(i).lat,
                    code_points.get(i).lng,
                    beginTile.x, beginTile.y);

            shapeRenderer.circle(marker.x, marker.y, 1);
        }
        shapeRenderer.end();

        // boat positions
        /*for(int i=0; i<boatAnimation.getInterpolatedPositions().length; i++){
            shapeRenderer.setProjectionMatrix(camera.combined);
            shapeRenderer.setColor(Color.RED);
            shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
            shapeRenderer.circle(boatAnimation.getInterpolatedPositions()[i].x, boatAnimation.getInterpolatedPositions()[i].y, 10);
            shapeRenderer.end();
        }*/
    }

    @Override
    public void dispose() {
        shapeRenderer.dispose();
        hudStage.dispose();
        geojson_points_mesh.dispose();
    }

    @Override
    public boolean touchDown(float x, float y, int pointer, int button) {
        touchPosition.set(x, y, 0);
        camera.unproject(touchPosition);
        return false;
    }

    @Override
    public boolean tap(float x, float y, int count, int button) {
        return false;
    }

    @Override
    public boolean longPress(float x, float y) {
        return false;
    }

    @Override
    public boolean fling(float velocityX, float velocityY, int button) {
        return false;
    }

    @Override
    public boolean pan(float x, float y, float deltaX, float deltaY) {
        camera.translate(-deltaX, deltaY);
        return false;
    }

    @Override
    public boolean panStop(float x, float y, int pointer, int button) {
        return false;
    }

    @Override
    public boolean zoom(float initialDistance, float distance) {
        if (initialDistance >= distance)
            camera.zoom += 0.02;
        else
            camera.zoom -= 0.02;
        return false;
    }

    @Override
    public boolean pinch(Vector2 initialPointer1, Vector2 initialPointer2, Vector2 pointer1, Vector2 pointer2) {
        return false;
    }

    @Override
    public void pinchStop() {

    }

    private void handleInput() {
        if (Gdx.input.isKeyPressed(Input.Keys.A)) {
            camera.zoom += 0.02;
        }
        if (Gdx.input.isKeyPressed(Input.Keys.Q)) {
            camera.zoom -= 0.02;
        }
        if (Gdx.input.isKeyPressed(Input.Keys.LEFT)) {
            camera.translate(-3, 0, 0);
        }
        if (Gdx.input.isKeyPressed(Input.Keys.RIGHT)) {
            camera.translate(3, 0, 0);
        }
        if (Gdx.input.isKeyPressed(Input.Keys.DOWN)) {
            camera.translate(0, -3, 0);
        }
        if (Gdx.input.isKeyPressed(Input.Keys.UP)) {
            camera.translate(0, 3, 0);
        }

        camera.zoom = MathUtils.clamp(camera.zoom, 0.5f, 2f);

        float effectiveViewportWidth = camera.viewportWidth * camera.zoom;
        float effectiveViewportHeight = camera.viewportHeight * camera.zoom;

        camera.position.x = MathUtils.clamp(camera.position.x, effectiveViewportWidth / 2f, Constants.MAP_WIDTH - effectiveViewportWidth / 2f);
        camera.position.y = MathUtils.clamp(camera.position.y, effectiveViewportHeight / 2f, Constants.MAP_HEIGHT - effectiveViewportHeight / 2f);
    }

    private Actor createButtons() {
        Table table = new Table();
        table.defaults().pad(20);

        TextButton langButton = new TextButton("Lang", skin, "toggle");
        langButton.addListener(new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                showLangExample = !showLangExample;
            }
        });

        TextButton animButton = new TextButton("Animation", skin);
        animButton.addListener(new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                stage.addActor(boatAnimation.create());
            }
        });

        TextButton quitButton = new TextButton("Quit", skin);
        quitButton.addListener(new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                Gdx.app.exit();
            }
        });

        Table buttonTable = new Table();
        buttonTable.defaults().padLeft(30).padRight(30);

        buttonTable.add(langButton).padBottom(15).expandX().fill().row();
        buttonTable.add(animButton).padBottom(15).fillX().row();
        buttonTable.add(quitButton).fillX();


        table.add(buttonTable);
        table.left();
        table.top();
        table.setFillParent(true);
        table.pack();

        return table;
    }
}
