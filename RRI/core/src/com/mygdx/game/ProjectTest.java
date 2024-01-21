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
import com.badlogic.gdx.scenes.scene2d.ui.TextArea;
import com.badlogic.gdx.scenes.scene2d.ui.TextButton;
import com.badlogic.gdx.scenes.scene2d.ui.Label;
import com.badlogic.gdx.scenes.scene2d.ui.ScrollPane;
import com.badlogic.gdx.scenes.scene2d.utils.ClickListener;
import com.badlogic.gdx.scenes.scene2d.utils.TextureRegionDrawable;
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
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.ui.Label;
import com.badlogic.gdx.scenes.scene2d.ui.ScrollPane;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.ui.TextButton;
import com.badlogic.gdx.scenes.scene2d.utils.ClickListener;
import com.badlogic.gdx.utils.Align;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

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
    ScrollPane scrollPane;
    private Stage hudStage;
    private Skin skin;
    private boolean showLangExample = false;
    private boolean animationStarted = false;
    Texture backgroundTexture;
    // animation
    private Stage stage;
    private FitViewport viewport;
    private Texture clickImage;
    private Texture spor;
    private Texture bomb;
    private Texture mark;
    private Texture marker;
    private Texture weather;
    private Texture raindrop;
    private boolean risiMarkerje=true;
    Array<Image> raindrops;


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
    boolean testdate=true;

    Map<Vector2, Integer> newsByLocation=new HashMap<>();

    List<Vector2> circleLocations = new ArrayList<>();
    List<List<INews>> razdeljeneNovice = new ArrayList<>();
    Table contentTable;
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
        backgroundTexture= new Texture(Gdx.files.internal("bacground.jpg"));
        //clickImage= new Texture(Gdx.files.internal("bomb.png"));
        code = GeolangKt.load();
        System.out.println(code);

        String geojson = GeolangKt.get_geojson_from_interpreter(code);
        System.out.println(geojson);

        code_points = GeolangKt.get_points_from_geojson(geojson);

        shapeRenderer = new ShapeRenderer();
        clickImage= new Texture(Gdx.files.internal("mark.png"));
        mark = new Texture(Gdx.files.internal("novicebasic.png"));
        bomb = new Texture(Gdx.files.internal("bomb.png"));
        marker = new Texture(Gdx.files.internal("marker.png"));
        spor = new Texture(Gdx.files.internal("spotr.png"));
        weather = new Texture(Gdx.files.internal("weather.png"));
        raindrop= new Texture(Gdx.files.internal("kaplaj.png"));

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
        //stage.addActor(boatAnimation.create());
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
        drawImageMarkers();
        hudStage.act(Gdx.graphics.getDeltaTime());
        stage.act(Gdx.graphics.getDeltaTime());
        hudStage.act(Gdx.graphics.getDeltaTime());
        hudStage.draw();
        if(risiMarkerje) {
            stage.draw();
        }

        drawMarkers();
        if(raindrops != null && raindrops.size > 0) {
            for (Image raindrop : raindrops) {
                if (!raindrop.hasActions()) {
                    raindrops.removeValue(raindrop, true);
                    raindrop.remove();
                }
            }
        }
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
    public static boolean hasCommonElement(List<String> list1, List<String> list2) {//NAJE CE SE NAHAJA STRING V LISTU
        for (String str : list1) {
            if (list2.contains(str)) {
                return true;
            }
        }
        return false;
    }
    int katagorijeMarkerjev(List<String>catagories){//NAJDE KATERA KATAGORIJA JE ZA SLIKO
        int rezultat=0;
        List<String> weatherCategories = new ArrayList<>(Arrays.asList("toča", "nevihta","okolje", "vreme", "dež", "megla", "sončno", "sneg", "sneženo", "ploha"));
        List<String> sportCategories = new ArrayList<>(Arrays.asList("rekreacija", "gibanje", "šport", "nogomet", "košarka", "sport", "tenis", "gimnastika", "jahanje", "smučanje", "smuk", "rokomet"));
        List<String> warCategories = new ArrayList<>(Arrays.asList("vojna", "napad", "bombandiranje", "tank"));
        if (hasCommonElement(catagories, weatherCategories)) {
            rezultat=1;
            // Do something related to weather
        } else if (hasCommonElement(catagories, sportCategories)) {
            rezultat=2;
            // Do something related to sports
        } else if(hasCommonElement(catagories, warCategories)){
            rezultat=3;
        }
        return rezultat;

    }

    private void drawImageMarkers(){//NARISE MARKERJE
        /*Map<Vector2, List<INews>> hashMapOfNews = new HashMap<>();
        List<List<INews>> newsWithLocations = new ArrayList<>();
        for (int i = 0; i < news.size(); i++) {
            Vector2 markerTexture = MapRasterTiles.getPixelPosition(
                    news.get(i).getLocation().getCoordinates().getSecond(),
                    news.get(i).getLocation().getCoordinates().getFirst(),
                    beginTile.x, beginTile.y);
            if(!hashMapOfNews.containsKey(markerTexture)){
                hashMapOfNews.put(markerTexture,new ArrayList<>());
                hashMapOfNews.get(markerTexture).add(news.get(i).)
            }

        }*/
        Map<Vector2, Integer> displayedMarkers=new HashMap<>();
        Collections.sort(news, new Comparator<INews>() {
            @Override
            public int compare(INews news1, INews news2) {
                return news2.getDate().compareTo(news1.getDate());
            }
        });

        for (int i = 0; i < news.size(); i++) {
            if(testdate) {
                System.out.println(news.get(i).getDate());
            }
            Image image;
            Geolocation singleGeolocation = new Geolocation(news.get(i).getLocation().getCoordinates().getSecond(), news.get(i).getLocation().getCoordinates().getFirst());
            Vector2 imagePosition = positionFromGeolocation(singleGeolocation, beginTile);
            int rezltatkatagorij=katagorijeMarkerjev(news.get(i).getCategories());
            if(rezltatkatagorij==1){
                image = new Image(weather);
            }
            else if(rezltatkatagorij==2){
                image = new Image(spor);
            }
            else if(rezltatkatagorij==2){
                image = new Image(bomb);
            }else {
                image= new Image(mark);
            }

            //image = new Image(clickImage);
            image.setWidth(50f);
            image.setHeight(50f);

                image.setPosition(imagePosition.x - 25, imagePosition.y - 25);


// Add the image to the stage
        if(!newsByLocation.containsKey(imagePosition)) {
            stage.addActor(image);
            newsByLocation.put(imagePosition,rezltatkatagorij);
        }
        }
        testdate=false;

    }

    private void drawMarkers() {//TU NE BRISI KAJ !!!!!!! IZRACUNA POZICIJE NA ZEMLJEVIDU, KI JIH LAHKO PRITISNES DA DOBIS NOVICE
        // DA NE POZABIM
        // IDEJA:
        // 1. "markerji" se izrisujejo kot kategorije npr. novice/web-app/src/assets/
        // 2. animacija: dež ki pada ob kategoriji "toča/dež..."
        //    pomoje zelo simpl, 3 kaplje narišeš pa sam uporabiš moveTo dol po y za ene 20 pisklov
        //    potem se pa to loopa
        circleLocations.clear();
        newsByLocation.clear();
        int vrednost=0;
        shapeRenderer.setProjectionMatrix(camera.combined);
        shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
        //spriteBatch.setProjectionMatrix(camera.combined);
        //spriteBatch.begin();
        for (int i = 0; i < news.size(); i++) {

            Vector2 marker = MapRasterTiles.getPixelPosition(
                    news.get(i).getLocation().getCoordinates().getSecond(),
                    news.get(i).getLocation().getCoordinates().getFirst(),
                    beginTile.x, beginTile.y);

            if (!newsByLocation.containsKey(marker)) {
                newsByLocation.put(marker, vrednost);
                vrednost++;
                circleLocations.add(marker);
            }

            if(!animationStarted) {
                //System.out.println("Hello, World!");
                System.out.println(news.get(i).getLocation().getCoordinates());
               // System.out.println(marker.x);
               // System.out.println(marker.y);
            }

            shapeRenderer.setColor(Color.GREEN);
            //shapeRenderer.circle(marker.x, marker.y, 20);
            //spriteBatch.draw(clickImage, marker.x, marker.y, 40, 40);
        }
        animationStarted=true;
        shapeRenderer.end();
        //spriteBatch.end();
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
    private Vector2 positionFromGeolocation(Geolocation geolocation, ZoomXY beginTile) {
        return MapRasterTiles.getPixelPosition(geolocation.lat, geolocation.lng, beginTile.x, beginTile.y);
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
    public boolean tap(float x, float y, int count, int button) {//PRITISK NA MAPO
        System.out.println("Tap: ");
        Vector3 worldCoordinates = camera.unproject(new Vector3(x, y, 0));

        // Call the handleClick method with world coordinates
        handleClick(worldCoordinates.x, worldCoordinates.y);

        return true;
    }

    @Override
    public boolean longPress(float x, float y) {
        return false;

    }
    public void handleClick(float clickX, float clickY) {//UREJANJE PRITISKA NA MAPO TUKAJ SE USTVARI PRIKAZ NOVIC
         ArrayList<INews> thisPointnews=new ArrayList<>();;

        // Check if the click is within the radius of any circle
        for (Vector2 circleLocation : circleLocations) {
            float distance = new Vector2(clickX, clickY).dst(circleLocation);

            // If the click is within the radius of the circle
            if (distance <= 20) {
                risiMarkerje=false;
                for (int i = 0; i < news.size(); i++) {

                    Vector2 marker = MapRasterTiles.getPixelPosition(
                            news.get(i).getLocation().getCoordinates().getSecond(),
                            news.get(i).getLocation().getCoordinates().getFirst(),
                            beginTile.x, beginTile.y);
                    float epsilon = 0.001f;
                    if(circleLocation.epsilonEquals(marker, epsilon)){
                        thisPointnews.add(news.get(i));
                    }

                }
                System.out.println(thisPointnews);
                System.out.println("Clicked on circle at: " + circleLocation);

                //Table newContent = new Table();
                contentTable.clear();
                Table titletable=new Table();
                Image imageNA;
                contentTable.add(new Label("NOVICE: ", skin)).left().row();
                for(int j = 0; j < thisPointnews.size(); j++) {
                    int rezltatkatagorij=katagorijeMarkerjev(thisPointnews.get(j).getCategories());
                    if(rezltatkatagorij==1){
                        imageNA = new Image(weather);
                    }
                    else if(rezltatkatagorij==2){
                        imageNA = new Image(spor);
                    }
                    else if(rezltatkatagorij==2){
                        imageNA = new Image(bomb);
                    }else {
                        imageNA= new Image(mark);
                    }

                    //image = new Image(clickImage);

                    Label label = new Label(thisPointnews.get(j).getTitle(), skin);
                    final int index = j;
                    label.addListener(new ClickListener() {//OB PRITISKU NA NOVICO SE POKAZE PRIKAZ NOVICE
                        @Override
                        public void clicked(InputEvent event, float x, float y) {
                            // Display the text of the clicked label
                            contentTable.clear();
                            int rezltatkatprikaz=katagorijeMarkerjev(thisPointnews.get(index).getCategories());
                            if(rezltatkatprikaz==1) {
                                rain(contentTable);
                            }
                            Table newContent = new Table();
                            newContent.add(new Label("TITLE:",skin)).left().row();
                            newContent.add(new Label(thisPointnews.get(index).getTitle(), skin)).left().row();
                            newContent.add(new Label("URL:",skin)).left().row();
                            int lengthURL=thisPointnews.get(index).getUrl().length();
                            int newlengthURL=lengthURL;
                            for (int i = 0; i < newlengthURL; i += 48) {
                                int end = Math.min(i + 48, newlengthURL);
                                String part = thisPointnews.get(index).getContent().substring(i, end);

                                Label labelContentPart = new Label(part, skin);
                                newContent.add(labelContentPart).left().row();;

                            }
                            //newContent.add(new Label(thisPointnews.get(index).getUrl(), skin)).left().row();
                            newContent.add(new Label("DATE:",skin)).left().row();
                            newContent.add(new Label(thisPointnews.get(index).getDate().toString(), skin)).left().row();
                            newContent.add(new Label("AUTHORD:",skin)).left().row();
                            newContent.add(new Label(thisPointnews.get(index).getAuthors().toString(), skin)).left().row();
                            newContent.add(new Label("CONTENT:",skin)).left().row();
                            int length=thisPointnews.get(index).getContent().length();
                            int newlength=length/2;
                            for (int i = 0; i < newlength; i += 48) {
                                int end = Math.min(i + 48, newlength);
                                String part = thisPointnews.get(index).getContent().substring(i, end);

                                Label labelContentPart = new Label(part, skin);
                               newContent.add(labelContentPart).left().row();;

                            }
                            newContent.add(new Label("CATEGORIES:",skin)).left().row();
                            newContent.add(new Label(thisPointnews.get(index).getCategories().toString(), skin)).left().row();
                            TextButton backbtn = new TextButton("Back to map", skin, "toggle");
                            backbtn.addListener(new ClickListener() {
                                @Override
                                public void clicked(InputEvent event, float x, float y) {
                                    contentTable.clear();
                                    risiMarkerje=true;
                                    contentTable.add(new Label("Tukaj bodo novice:",skin));
                                }
                            });
                            //contentTable.clear();
                            //contentTable=newContent;
                            newContent.add(backbtn).left().row();
                            ScrollPane scrollPane1 = new ScrollPane(newContent);
                            contentTable.add(scrollPane1);
                            TextureRegionDrawable drawable = new TextureRegionDrawable(backgroundTexture);
                            contentTable.background(drawable);


                            //scrollPane.clear();
                            //scrollPane.setScrollingDisabled(false, true);
                            //scrollPane.setActor(newContent);
                            //scrollPane.layout();

                        }
                    });

                    titletable.add(label).left().pad(0).space(0);
                    titletable.add(imageNA).left().size(30f,30f).pad(0).space(0).row();

                }
                TextButton backbtn2 = new TextButton("Back to map", skin, "toggle");
                backbtn2.addListener(new ClickListener() {
                    @Override
                    public void clicked(InputEvent event, float x, float y) {
                        contentTable.clear();
                        risiMarkerje=true;
                        contentTable.add(new Label("Tukaj bodo novice:",skin));
                    }
                });
                titletable.add(backbtn2).left().row();
                titletable.left();
                ScrollPane scrollPane2 = new ScrollPane(titletable);
                contentTable.add(scrollPane2);
                TextureRegionDrawable drawable = new TextureRegionDrawable(backgroundTexture);
                contentTable.background(drawable);

                //contentTable=newContent;
                //scrollPane.clear();
                //scrollPane.setScrollingDisabled(false, true);
                //scrollPane.setActor(newContent);
                //scrollPane.layout();

                // Perform any actions you want here
            }
        }
    }
    void rain(Table containerTable){//ANIMACIJA ZA DEZ
        Image raindropImage = new Image(raindrop);
        float initialX;
        float initialY = Gdx.graphics.getHeight();
        int numRaindrops = 25;
        float screenWidth = Gdx.graphics.getWidth();
        raindrops = new Array<>();
        float spacing = 20f;
        float duration=5f;
        raindropImage.setSize(20f,20f);
        // Create raindrops and add them to the stage
        for (int row = 0; row < 10; row++) {
            for (int i = 0; i < numRaindrops; i++) {
                Image raindrop = new Image(raindropImage.getDrawable());

                // Set initial position
                //initialX = i * (raindropImage.getWidth() + spacing);

                // Create a new Image using the raindrop texture
                float randomX = MathUtils.random(screenWidth);
                raindrop.setSize(20f, 20f);
                // Set the initial position of the raindrop
                raindrop.setPosition(randomX, initialY);

                // Create a sequence of actions for each raindrop
                raindrop.addAction(Actions.sequence(
                        Actions.moveTo(randomX, 0, duration),
                        Actions.removeActor()
                ));

                // Add raindrop to the stage and array
                hudStage.addActor(raindrop);
                raindrops.add(raindrop);
                containerTable.addActor(raindrop);
            }
            initialY += 20f + spacing;
            duration+=0.5f;
        }

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
                //showLangExample = !showLangExample;
                //rain();
            }
        });
        langButton.left();

        TextButton animButton = new TextButton("Legend", skin);
        animButton.addListener(new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                contentTable.clear();
                risiMarkerje=false;
                //stage.addActor(boatAnimation.create());
                //Table newContent = new Table();
                contentTable.add(new Label("LEGEND:", skin)).left().row();
                contentTable.add(new Label("Typs of news created last on location:", skin)).left().row();
                Table legendTable=new Table();
                Image bombaL = new Image(bomb);
                //bombaL.setWidth(50f);
                //bombaL.setHeight(50f);
                legendTable.add(bombaL).left().size(30f,30f).pad(0).space(0);
                legendTable.add(new Label("- War news", skin)).left().pad(0).space(0).row();
                Image wetherL = new Image(weather);
                //wetherL.setWidth(50f);
                //wetherL.setHeight(50f);
                legendTable.add(wetherL).left().size(30f,30f).pad(0).space(0);
                legendTable.add(new Label("- Wether/environment news", skin)).left().pad(0).space(0).row();
                Image sportL = new Image(spor);
                //sportL.setWidth(50f);
                //sportL.setHeight(50f);
                legendTable.add(sportL).left().size(30f,30f).pad(0).space(0);
                legendTable.add(new Label("- Sport news", skin)).left().pad(0).space(0).row();
                Image basicL = new Image(mark);
                //basicL.setWidth(50f);
                //basicL.setHeight(50f);
                legendTable.add(basicL).left().size(30f,30f).pad(0).space(0);
                legendTable.add(new Label("- Oather news", skin)).left().pad(0).space(0).row();
                TextButton backbtnL =new TextButton("Back to map", skin);
                backbtnL.addListener(new ClickListener() {
                    @Override
                    public void clicked(InputEvent event, float x, float y) {
                        contentTable.clear();
                        risiMarkerje=true;
                        contentTable.add(new Label("Tukaj bodo novice:",skin));
                    }
                });
                contentTable.add(legendTable).left().row();
                contentTable.add(backbtnL).left().row();


            }
        });
        animButton.left();
        TextButton quitButton = new TextButton("Quit", skin);
        quitButton.addListener(new ClickListener() {
            @Override
            public void clicked(InputEvent event, float x, float y) {
                Gdx.app.exit();
            }
        });
        quitButton.left();
        Table buttonTable = new Table();
        buttonTable.defaults().padLeft(30).padRight(30);

        buttonTable.add(langButton).padBottom(15).expandX().fill().row();
        buttonTable.add(animButton).padBottom(15).fillX().row();
        buttonTable.add(quitButton).fillX();

        contentTable = new Table();
        contentTable.defaults().pad(10);
        //Image image = new Image(clickImage);
        //contentTable.add(image).left().row();
        TextureRegionDrawable drawable = new TextureRegionDrawable(backgroundTexture);
        contentTable.add(new Label("Tukaj bodo novice:",skin));
        contentTable.background(drawable);
        buttonTable.background(drawable);

        // Add a lot of information to the content table

        /*for (int i = 0; i < 50; i++) {
            final String labelText = "Information " + i;
            Label label = new Label(labelText, skin);
            label.addListener(new ClickListener() {
                @Override
                public void clicked(InputEvent event, float x, float y) {
                    // Display the text of the clicked label
                    Gdx.app.log("LabelClick", "Label clicked! Text: " + labelText);
                }
            });
            contentTable.add(label).row();
        }*/

        // Create a ScrollPane and set its widget to the content table
        //scrollPane = new ScrollPane(contentTable);

        table.add(buttonTable).row();
        table.add(contentTable);
        table.left();
        table.top();
        table.setFillParent(true);
        table.pack();
        table.setDebug(true);

        return table;
    }
}
