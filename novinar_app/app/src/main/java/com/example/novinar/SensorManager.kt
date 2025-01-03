package com.example.novinar

import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Geocoder
import android.location.Location
import android.location.LocationManager
import android.os.SystemClock
import java.text.SimpleDateFormat
import java.util.*

object SensorManager : SensorEventListener {
    private val sensorData = mutableMapOf<String, Any>()
    private var accelerometerData = Triple(0f, 0f, 0f)
    private var pressureData = 0f
    private var temperatureData = 0f

    private val sensorManager = App.context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    init {
        registerSensors()
    }

    private fun registerSensors() {
        // Accelerometer
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }

        // Pressure
        val pressureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
        pressureSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }


    }

    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            when (it.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    accelerometerData = Triple(it.values[0], it.values[1], it.values[2])
                }
                Sensor.TYPE_PRESSURE -> {
                    pressureData = it.values[0]
                }
                Sensor.TYPE_AMBIENT_TEMPERATURE -> {
                    temperatureData = it.values[0]
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed for this implementation
    }
    private fun getPhoneTemperature(): Double {
        // Use ambient temperature if available
        if (temperatureData != 0f) {
            return temperatureData.toDouble()
        }

        // Fallback to battery temperature
        val intent = App.context.registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val batteryTemp = intent?.getIntExtra("temperature", -1) ?: -1
        return if (batteryTemp != -1) batteryTemp / 10.0 else Double.NaN // Convert to Celsius
    }

    fun updateSensors() {
        // Location
        val location = getLocation()
        sensorData["latitude"] = location?.latitude ?: "N/A"
        sensorData["longitude"] = location?.longitude ?: "N/A"

        // Approximate location
        val address = location?.let { getApproximateLocation(it) }
        sensorData["town"] = address?.locality ?: "Unknown"
        sensorData["country"] = address?.countryName ?: "Unknown"

        // Time
        val timestamp = SystemClock.elapsedRealtime()
        sensorData["timestamp"] = convertTimestampToDateTime(timestamp)

        // Internal Temperature
        sensorData["Phone temp"] = getPhoneTemperature()

        // Accelerometer
        sensorData["accelerometer"] = "X: ${accelerometerData.first}, Y: ${accelerometerData.second}, Z: ${accelerometerData.third}"

        // Pressure
        sensorData["pressure"] = pressureData

    }

    fun getSensorData(): Map<String, Any> {
        return sensorData
    }

    private fun getLocation(): Location? {
        val locationManager = App.context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        return try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            } else if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            } else {
                null
            }
        } catch (e: SecurityException) {
            e.printStackTrace()
            null
        }
    }

    private fun getApproximateLocation(location: Location): android.location.Address? {
        return try {
            val geocoder = Geocoder(App.context, Locale.getDefault())
            val addresses = geocoder.getFromLocation(location.latitude, location.longitude, 1)
            addresses?.firstOrNull()
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun convertTimestampToDateTime(timestamp: Long): String {
        val date = Date(System.currentTimeMillis() + timestamp - SystemClock.elapsedRealtime())
        val format = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        return format.format(date)
    }
}
