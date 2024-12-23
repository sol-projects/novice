package io.github.some_example_name.oop;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.utils.Pool;
import com.badlogic.gdx.graphics.g2d.TextureAtlas.AtlasRegion;
public class SpeedBoost extends GameObject implements Pool.Poolable {
    //private static final Texture speedBoostTexture = new Texture("speedBoost.png");

    public SpeedBoost(AtlasRegion speedBoostTexture) {
        super(speedBoostTexture);
        bounds.width = speedBoostTexture.getRegionWidth() * 4f; // Apply scaling if needed
        bounds.height = speedBoostTexture.getRegionHeight() * 4f;
    }



    @Override
    public void update(float delta) {
        bounds.y -= 150 * delta;
    }

    @Override
    public void reset() {
        bounds.setPosition(0, 0);
    }
}
