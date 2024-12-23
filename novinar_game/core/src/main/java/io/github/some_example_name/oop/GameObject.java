package io.github.some_example_name.oop;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.math.Rectangle;
import com.badlogic.gdx.utils.Pool;
import com.badlogic.gdx.graphics.g2d.TextureAtlas.AtlasRegion;
import com.badlogic.gdx.math.Rectangle;
public abstract class GameObject implements Pool.Poolable {
    protected AtlasRegion texture;
    protected Rectangle bounds;

    public GameObject(AtlasRegion textureRegion) {
        this.texture = textureRegion;
        this.bounds = new Rectangle(0, 0, textureRegion.getRegionWidth(), textureRegion.getRegionHeight());
    }

    public Rectangle getBounds() {
        return bounds;
    }

    public AtlasRegion getTexture() {
        return texture;
    }
    public void setPosition(float x, float y) {
        bounds.setPosition(x, y);
    }


    public boolean collisionCheck(GameObject other) {
        return this.bounds.overlaps(other.getBounds());
    }

    public abstract void update(float delta);

    public abstract void reset();
}
