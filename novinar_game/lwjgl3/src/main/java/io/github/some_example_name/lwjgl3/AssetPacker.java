package io.github.some_example_name.lwjgl3;


import com.badlogic.gdx.tools.texturepacker.TexturePacker;

public class AssetPacker {
    private static final String RAW_ASSETS_PATH = "assets-raw";
    private static final String PACKED_ASSETS_PATH = "assets";
    private static final String PACK_FILE = "game";

    public static void main(String[] args) {
        TexturePacker.Settings settings = new TexturePacker.Settings();
        settings.maxWidth = 2048;
        settings.maxHeight = 2048;
        settings.combineSubdirectories = true;

        TexturePacker.process(settings, RAW_ASSETS_PATH, PACKED_ASSETS_PATH, PACK_FILE);
        System.out.println("Atlas successfully created!");
    }
}
