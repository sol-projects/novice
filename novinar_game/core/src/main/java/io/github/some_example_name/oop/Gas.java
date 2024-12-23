package io.github.some_example_name.oop;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.utils.Pool;

import com.badlogic.gdx.graphics.g2d.TextureAtlas.AtlasRegion;

public class Gas extends GameObject {
    public Gas(AtlasRegion textureRegion) {
        super(textureRegion);
        bounds.width = textureRegion.getRegionWidth() /4f;
        bounds.height = textureRegion.getRegionHeight() /4f;
    }

    @Override
    public void update(float delta) {
        bounds.y -= 150 * delta;
    }

    @Override
    public void reset() {

    }
}

