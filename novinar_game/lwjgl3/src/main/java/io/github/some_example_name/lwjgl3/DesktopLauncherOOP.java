package io.github.some_example_name.lwjgl3;

import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;

import io.github.some_example_name.oop.MainOOP;

public class DesktopLauncherOOP {
    public static void main (String[] arg) {
        Lwjgl3ApplicationConfiguration config = new Lwjgl3ApplicationConfiguration();
        config.setTitle("OOP Version of the Game");
        config.setWindowedMode(800, 480);
        config.setForegroundFPS(60);
        new Lwjgl3Application(new MainOOP(), config);
    }
}
