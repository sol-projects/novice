package io.github.some_example_name.oop;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Vector2;

public class MapMarker {
    private Vector2 position;
    private Texture texture;

    public MapMarker(float x, float y, Texture texture) {
        this.position = new Vector2(x, y);
        this.texture = texture;
    }

    public void render(SpriteBatch batch) {
        batch.draw(texture, position.x, position.y, 32, 32); // Velikost markerja: 32x32 px
    }

    public Vector2 getPosition() {
        return position;
    }
}
