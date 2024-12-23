package com.example.novinar

import android.app.Application

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        private lateinit var instance: App

        val context get() = instance.applicationContext
    }
}
