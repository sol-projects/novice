package io.github.some_example_name.oop;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.graphics.g2d.TextureAtlas.AtlasRegion;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;

public class Car extends GameObject {
    private static final float MAX_SPEED_X = 500f;
    private static final float MAX_SPEED_Y = 300f;
    private static final float ACCELERATION = 300f;
    private static final float FRICTION = 100f;
    private boolean isSpeedBoosted = false;
    private float speedBoostDuration = 10f;
    private float speedBoostTimer = 0f;

    private float speedX = 0f;
    private float speedY = 0f;
    public int health = 100;
    public float gasRemaining = 100;

    public Car(AtlasRegion carTexture) {
        super(carTexture);
        bounds.width = carTexture.getRegionWidth() * 4f; // Apply scaling if needed
        bounds.height = carTexture.getRegionHeight() * 4f;
        bounds.x = Gdx.graphics.getWidth() / 2 - bounds.width / 2;
        bounds.y = 20;
    }

    @Override
    public void update(float delta) {
        if (isSpeedBoosted) {
            speedBoostTimer -= delta;
            if (speedBoostTimer <= 0) {
                isSpeedBoosted = false;
                resetSpeed();
            }
        }
        handleInput(delta);
        bounds.x += speedX * delta;
        bounds.y += speedY * delta;

        // Ensure the car stays within screen boundaries
        if (bounds.x < 0) {
            bounds.x = 0;
            speedX = 0;
        }
        if (bounds.x > Gdx.graphics.getWidth() - bounds.width) {
            bounds.x = Gdx.graphics.getWidth() - bounds.width;
            speedX = 0;
        }
        if (bounds.y < 0) {
            bounds.y = 0;
            speedY = 0;
        }
        if (bounds.y > Gdx.graphics.getHeight() - bounds.height) {
            bounds.y = Gdx.graphics.getHeight() - bounds.height;
            speedY = 0;
        }
    }

    private void handleInput(float delta) {
        float effectiveAcceleration = isSpeedBoosted ? ACCELERATION * 3 : ACCELERATION;
        float effectiveMaxXSpeed = isSpeedBoosted ? MAX_SPEED_X * 2 : MAX_SPEED_X;
        float effectiveMaxYSpeed = isSpeedBoosted ? MAX_SPEED_Y * 2 : MAX_SPEED_Y;

        if (Gdx.input.isKeyPressed(Input.Keys.LEFT)) {
            speedX -= effectiveAcceleration * delta;
        }
        if (Gdx.input.isKeyPressed(Input.Keys.RIGHT)) {
            speedX += effectiveAcceleration * delta;
        }
        if (Gdx.input.isKeyPressed(Input.Keys.UP)) {
            speedY += effectiveAcceleration * delta;
        }
        if (Gdx.input.isKeyPressed(Input.Keys.DOWN)) {
            speedY -= effectiveAcceleration * delta;
        }

        // Limit speed to maximum allowed values
        speedX = Math.max(-effectiveMaxXSpeed, Math.min(speedX, effectiveMaxXSpeed));
        speedY = Math.max(-effectiveMaxYSpeed, Math.min(speedY, effectiveMaxYSpeed));

        applyFriction(delta);
    }

    private void resetSpeed() {
        speedX = Math.signum(speedX) * (Math.abs(speedX) / 2);
        speedY = Math.signum(speedY) * (Math.abs(speedY) / 2);
    }

    public void activateSpeedBoost() {
        isSpeedBoosted = true;
        speedBoostTimer = speedBoostDuration;
    }

    private void applyFriction(float delta) {
        if (speedX != 0) {
            speedX += (speedX > 0 ? -FRICTION : FRICTION) * delta;
            if (Math.abs(speedX) < FRICTION * delta) {
                speedX = 0;
            }
        }
        if (speedY != 0) {
            speedY += (speedY > 0 ? -FRICTION : FRICTION) * delta;
            if (Math.abs(speedY) < FRICTION * delta) {
                speedY = 0;
            }
        }
    }

    @Override
    public void reset() {
        bounds.x = Gdx.graphics.getWidth() / 2 - bounds.width / 2;
        bounds.y = 20;
        isSpeedBoosted = false;
        speedX = 0;
        speedY = 0;
        health = 100;
        gasRemaining = 100;
    }
}
