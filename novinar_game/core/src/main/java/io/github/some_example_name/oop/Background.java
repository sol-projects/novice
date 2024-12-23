package io.github.some_example_name.oop;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;import com.badlogic.gdx.graphics.g2d.TextureAtlas.AtlasRegion;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;

public class Background {
    private AtlasRegion texture;
    private float y1, y2;
    private static final float SCROLL_SPEED = 150f;

    public Background(AtlasRegion texture) {
        this.texture = texture;
        y1 = 0;
        y2 = texture.getRegionHeight();
    }

    public void update(float delta) {
        y1 -= SCROLL_SPEED * delta;
        y2 -= SCROLL_SPEED * delta;

        if (y1 + texture.getRegionHeight() < 0) {
            y1 = y2 + texture.getRegionHeight();
        }
        if (y2 + texture.getRegionHeight() < 0) {
            y2 = y1 + texture.getRegionHeight();
        }
    }

    public void draw(SpriteBatch batch) {
        batch.draw(texture, 0, y1, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        batch.draw(texture, 0, y2, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
    }

    public void dispose() {
        texture.getTexture().dispose();
    }
}
