package io.github.some_example_name.Acceleration;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.math.Vector2;

public class Ball {
    private Vector2 position;
    private Vector2 velocity;
    private float radius;
    private Color color;
    private static final float GRAVITY = 500;
    private static final float BOUNCE_DAMPING = 0.7f;

    public Ball(float x, float y, float radius) {
        this.position = new Vector2(x, y);
        this.velocity = new Vector2(0, 0);
        this.radius = radius;
        this.color = new Color((float) Math.random(), (float) Math.random(), (float) Math.random(), 1);
    }

    public void update(float delta) {
        velocity.y -= GRAVITY * delta;

        position.add(velocity.cpy().scl(delta));

        if (position.y - radius <= 0) {
            position.y = radius;
            velocity.y = -velocity.y * BOUNCE_DAMPING;
        }
    }

    public void draw(ShapeRenderer shapeRenderer) {
        shapeRenderer.setColor(color);
        shapeRenderer.circle(position.x, position.y, radius);
    }
}
