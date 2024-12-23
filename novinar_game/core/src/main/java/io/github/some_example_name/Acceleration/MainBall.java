package io.github.some_example_name.Acceleration;

import com.badlogic.gdx.ApplicationAdapter;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.input.GestureDetector;

import java.util.ArrayList;
import java.util.List;


public class MainBall extends ApplicationAdapter {
    private ShapeRenderer shapeRenderer;
    private List<Ball> balls;

    @Override
    public void create() {
        shapeRenderer = new ShapeRenderer();
        balls = new ArrayList<>();

        Gdx.input.setInputProcessor(new GestureDetector(new GestureDetector.GestureAdapter() {
            @Override
            public boolean tap(float x, float y, int count, int button) {
                float radius = (float) (Math.random() * 20 + 10);
                Ball ball = new Ball(x, Gdx.graphics.getHeight() - y, radius);
                balls.add(ball);
                return true;
            }
        }));
    }

    @Override
    public void render() {
        Gdx.gl.glClearColor(1, 1, 1, 1);
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

        // Update and draw balls
        for (Ball ball : balls) {
            ball.update(Gdx.graphics.getDeltaTime());
        }

        // Drawing the balls
        shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
        for (Ball ball : balls) {
            ball.draw(shapeRenderer);
        }
        shapeRenderer.end();
    }

    @Override
    public void dispose() {
        shapeRenderer.dispose();
    }
}
