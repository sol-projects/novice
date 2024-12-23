package io.github.some_example_name.oop;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.utils.Pool;
import com.badlogic.gdx.graphics.g2d.TextureAtlas.AtlasRegion;

public class RoadSign extends GameObject implements Pool.Poolable {
    //private static final Texture roadSignTexture = new Texture("Mini Pixel Pack 2/Props/Misc_prop_road_block.png");

    public RoadSign(AtlasRegion roadSignTexture) {
        super(roadSignTexture);
        bounds.width = roadSignTexture.getRegionWidth() * 4f; // Apply scaling if needed
        bounds.height = roadSignTexture.getRegionHeight() * 4f;
    }

    @Override
    public void update(float delta) {
        bounds.y -= 150 * delta; // Move downward
    }

    @Override
    public void reset() {
        bounds.setPosition(0, 0);
    }
}
