import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    kotlin("multiplatform")
    id("org.jetbrains.compose")
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
