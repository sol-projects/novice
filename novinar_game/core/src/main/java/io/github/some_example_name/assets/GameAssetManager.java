package io.github.some_example_name.assets;

import com.badlogic.gdx.assets.AssetManager;

import com.badlogic.gdx.assets.AssetManager;

public class GameAssetManager {
    private static GameAssetManager instance;
    private AssetManager assetManager;

    private GameAssetManager() {
        assetManager = new AssetManager();
    }

    public static synchronized GameAssetManager getInstance() {
        if (instance == null) {
            instance = new GameAssetManager();
        }
        return instance;
    }

    public AssetManager getAssetManager() {
        return assetManager;
    }

    public void loadAllAssets() {
        assetManager.load(AssetDescriptors.ATLAS);
        assetManager.load(AssetDescriptors.PARTICLE_BLOWBACK);
        assetManager.load(AssetDescriptors.PARTICLE_TIRE);
        assetManager.finishLoading();
    }

    public void dispose() {
        assetManager.dispose();
    }
}
