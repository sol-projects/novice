plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    id("kotlin-parcelize")
}

android {
    namespace = "com.example.novinar"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.novinar"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures {
        dataBinding = true
        viewBinding = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {
    // MQTT Dependencies
    implementation ("org.eclipse.paho:org.eclipse.paho.client.mqttv3:1.2.5")
    implementation ("org.eclipse.paho:org.eclipse.paho.android.service:1.1.1")
    implementation ("com.google.android.gms:play-services-maps:18.1.0")
    // Retrofit
    implementation ("com.squareup.retrofit2:retrofit:2.9.0")
    implementation ("com.squareup.retrofit2:converter-gson:2.9.0")

    // AndroidX and Material
    implementation ("androidx.core:core-ktx:1.12.0")
    implementation( libs.androidx.appcompat)
            implementation (libs.material)
            implementation (libs.androidx.activity)
            implementation (libs.androidx.constraintlayout)

            // Navigation Components
            implementation (libs.androidx.navigation.fragment)
            implementation (libs.androidx.navigation.fragment.ktx)
            implementation (libs.androidx.navigation.ui.ktx)

            // Gson
            implementation ("com.google.code.gson:gson:2.8.9")

    // OSM Droid
    implementation ("org.osmdroid:osmdroid-android:6.1.15")
    implementation ("com.github.MKergall:osmbonuspack:6.9.0")

    // Google Maps and Location
    implementation ("com.google.android.gms:play-services-location:21.0.1")
    implementation ("com.google.android.gms:play-services-maps:18.1.0")

    // Faker for Dummy Data ds
    implementation ("com.github.javafaker:javafaker:1.0.2")
//
    // Testing
    testImplementation( libs.junit)
            androidTestImplementation (libs.androidx.junit)
            androidTestImplementation (libs.androidx.espresso.core)
}
