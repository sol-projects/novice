import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    kotlin("multiplatform")
    id("org.jetbrains.compose")
    kotlin("plugin.serialization")
}

group = "com.example"
version = "1.0-SNAPSHOT"

repositories {
    google()
    mavenCentral()
    maven("https://maven.pkg.jetbrains.space/public/p/compose/dev")
}

kotlin {
    jvm {
        jvmToolchain(11)
        withJava()
    }
    sourceSets {
        val jvmMain by getting {
            dependencies {
                implementation(compose.desktop.currentOs)
                implementation("org.seleniumhq.selenium:selenium-java:4.+")
                implementation("io.github.bonigarcia:webdrivermanager:5.+")
                implementation ("com.fasterxml.jackson.core:jackson-databind:2.13.0")
                implementation ("com.fasterxml.jackson.module:jackson-module-kotlin:2.13.0")
                implementation ("com.squareup.okhttp3:okhttp:4.9.1")
                implementation ("com.google.code.gson:gson:2.8.9")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.3.2")
                implementation ("org.jetbrains.kotlinx:kotlinx-serialization-json:1.2.1")
                implementation("org.json:json:20210307")
                implementation ("org.mongodb:mongodb-driver-sync:4.3.1")


            }
        }
        val jvmTest by getting
    }
}

compose.desktop {
    application {
        mainClass = "MainKt"
        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "desktop-app"
            packageVersion = "1.0.0"
        }
    }
}
