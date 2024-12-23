package io.github.some_example_name.assets;

import com.badlogic.gdx.assets.AssetDescriptor;
import com.badlogic.gdx.files.FileHandle;
import com.badlogic.gdx.graphics.g2d.ParticleEffect;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.assets.AssetDescriptor;
import com.badlogic.gdx.graphics.g2d.ParticleEffect;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;

public class AssetDescriptors {
    public static final AssetDescriptor<TextureAtlas> ATLAS =
        new AssetDescriptor<>(AssetPaths.ATLAS, TextureAtlas.class);

    public static final AssetDescriptor<ParticleEffect> PARTICLE_BLOWBACK =
        new AssetDescriptor<>(AssetPaths.PARTICLE_BLOWBACK, ParticleEffect.class);

    public static final AssetDescriptor<ParticleEffect> PARTICLE_TIRE =
        new AssetDescriptor<>(AssetPaths.PARTICLE_TIRE, ParticleEffect.class);

    private AssetDescriptors() {
    }
}
