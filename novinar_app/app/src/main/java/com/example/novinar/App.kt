package com.example.novinar

import android.app.Application
import android.os.Handler
import android.os.Looper

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this

        startSensorUpdates()
    }

    private fun startSensorUpdates() {
        val handler = Handler(Looper.getMainLooper())
        val updateInterval = 10000L // 10 sec

        handler.post(object : Runnable {
            override fun run() {
                SensorManager.updateSensors()
                handler.postDelayed(this, updateInterval)
            }
        })
    }

    companion object {
        private lateinit var instance: App
        val context get() = instance.applicationContext
    }
}
