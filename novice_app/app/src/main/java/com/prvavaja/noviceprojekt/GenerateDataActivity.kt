package com.prvavaja.noviceprojekt

import android.app.Application
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.prvavaja.noviceprojekt.databinding.ActivityGenerateDataBinding
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.SensorManager
import org.osmdroid.util.GeoPoint
import org.osmdroid.api.IMapController
import android.hardware.Sensor
import android.os.Handler
import android.os.Looper
import android.hardware.SensorEventListener
import android.hardware.SensorEvent
import android.location.Location
import androidx.annotation.RequiresApi
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import io.realm.kotlin.ext.realmListOf
import io.realm.kotlin.types.RealmInstant
import kotlinx.coroutines.runBlocking
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

class GenerateDataActivity : AppCompatActivity() {
    private lateinit var binding: ActivityGenerateDataBinding
    private lateinit var mapController: IMapController
    private var clickedLocation: GeoPoint? = null
    private var isLocationSelected: Boolean = false
    lateinit var handler1: Handler
    lateinit var handler2: Handler
    lateinit var handler3: Handler
    lateinit var handler4: Handler
    lateinit var myApplication: MyAplication
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        myApplication = application as MyAplication
        //Configuration.getInstance().load(applicationContext,this.getPreferences(Context.MODE_PRIVATE))
        binding = ActivityGenerateDataBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val switch1=binding.temperatureSwitchSimulated
        val switch2=binding.temperatureSwitchSensor
        val switch3=binding.humidSwitchSimulated
        val switch4=binding.humidSwitchSensor

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        handler1 = Handler(Looper.getMainLooper())
        handler2 = Handler(Looper.getMainLooper())
        handler3 = Handler(Looper.getMainLooper())
        handler4 = Handler(Looper.getMainLooper())

        /*mySwitch.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                // Switch is ON
                mySwitch.text = "Switch is ON"
            } else {
                // Switch is OFF
                mySwitch.text = "Switch is OFF"
            }
        }*/

        if (ActivityCompat.checkSelfPermission(
                this,
                android.Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            fusedLocationClient.lastLocation
                .addOnSuccessListener { location: Location? ->
                    if (location != null) {
                        realLastLocation.coordinates = Pair(location.longitude, location.latitude)
                    } else {
                    }
                }
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.ACCESS_FINE_LOCATION),
                REQUEST_CODE_LOCATION_PERMISSION
            )

            fusedLocationClient.lastLocation
                .addOnSuccessListener { location: Location? ->
                    if (location != null) {
                        realLastLocation.coordinates = Pair(location.longitude, location.latitude)
                    } else {
                    }
                }
        }

        if (hasAmbientTemperatureSensor()) {
            println("ima ambient senzor")
        } else {
            println("nima ambient senzor")
        }
        if (hasHumiditySensor()) {
            println("ima vlaznostni senzor")
        } else {
            println("nima vlaznostni senzor")
        }
        var temperatura = 0.0F
        var vlaznost = 0.0F
        //lateinit var simulatedTempHandler: Handler
        //simulatedTempHandler = Handler(Looper.getMainLooper())

        //val toggleTemperature: Switch = binding.temperatureSwitchSimulated
        switch1.setOnCheckedChangeListener { _, isChecked ->

            val updateSimulatedTemp = object : Runnable {
                override fun run() {

                    //Simulirana temperatura
                    val tempFrom = binding.cardTemperaturaSIntervalEdittextFrom.text.toString().toInt()
                    val tempTo = binding.cardTemperaturaSIntervalEdittextTo.text.toString().toInt()
                    val interval = binding.cardTemperaturaSIntervalEdittextInterval.text.toString().toLong() * 1000
                    val simulatedTemp = (tempFrom..tempTo).random()

                    //Simulirana lokacija
                    //val lokacija = binding.adressInput.text.toString()

                    println("Temperatura: $simulatedTemp")
                    //println("Lokacija: " + lokacija)

                    //createurrentTempInDataBase(simulatedTemp, lokacija)
                    handler1.postDelayed(this, interval)

                }
            }

            if (isChecked) {
                switch1.text ="Running"
                handler1.post(updateSimulatedTemp)
            }
            else{
                switch1.text ="Disabled"
                handler1.removeMessages(0);
            }

        }


        val sensorManagerTemperature = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        var list = sensorManagerTemperature.getSensorList(Sensor.TYPE_AMBIENT_TEMPERATURE)
        var sensorEvent = object: SensorEventListener{
            override fun onSensorChanged(sensorEvent: SensorEvent?) {
                val values = sensorEvent?.values
                if (values != null && values[0] > -1.0E29) {
                    var temp = values[0]
                    println(temp)
                    temperatura=temp
                    writeTempToServer(temperatura)
                }
            }
            override fun onAccuracyChanged(p0: Sensor?, p1: Int) {}
        }
        switch2.setOnCheckedChangeListener{_,isChecked->
            val updateSensorTemperature = object : Runnable {
                @RequiresApi(Build.VERSION_CODES.O)
                override fun run() {
                    runOnUiThread {
                        val cas = binding.cardTimeSensorDisplaySecondsInterval.text.toString().toInt()
                        println("Temperatura iz senozrja: $temperatura")
                        runBlocking {
                            myApplication.realm.write {
                                var article = NewsArticleRealm()
                                article.authors = realmListOf("senzor")
                                article.date = RealmInstant.now()
                                article.categories = realmListOf("senzor-temperatura")
                                article.content = temperatura.toString()
                                article.title = "Temperatura iz senzorja"
                                article.location = LocationRealm()
                                article.location!!.type = realLastLocation.type
                                println(realLastLocation.coordinates)
                                article.location!!.coordinates = realmListOf(realLastLocation.coordinates.first, realLastLocation.coordinates.second)
                                copyToRealm(article)
                            }
                        }

                        handler2.postDelayed(this, cas * 1000L)
                    }
                }
            }
            if (isChecked) {
                print("swithc2")
                sensorManagerTemperature.registerListener(sensorEvent, list.get(0), SensorManager.SENSOR_DELAY_NORMAL)
                handler2.post(updateSensorTemperature)
                switch2.text ="Running"
            }
            else{
                print("swithc2")
                handler2.removeMessages(0);
                switch2.text ="Disabled"
            }
        }

        switch3.setOnCheckedChangeListener { _, isChecked ->
            val updateSimulatedTemp = object : Runnable {
                override fun run() {
                    val humidFrom = binding.cardHumidSIntervalEdittextFrom.text.toString().toInt()
                    val humidTo = binding.cardHumidSIntervalEdittextTo.text.toString().toInt()
                    val cas = binding.cardHumidTimeSimDisplaySecondsEdit.text.toString().toLong() * 1000
                    val simulatedHumid = (humidFrom..humidTo).random()

                    //Simulirana lokacija
                    //val lokacija = binding.adressInput.text.toString()

                    println("Vlaznost: $simulatedHumid")
                    //println("Lokacija: " + lokacija)

                    //createurrentTempInDataBase(simulatedTemp, lokacija)
                    handler3.postDelayed(this, cas)

                }
            }

            if (isChecked) {
                switch3.text ="Running"
                handler3.post(updateSimulatedTemp)
            }
            else{
                switch3.text ="Disabled"
                handler3.removeMessages(0);
            }

        }


        val sensorManagerHumid = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        var list2 = sensorManagerHumid.getSensorList(Sensor.TYPE_RELATIVE_HUMIDITY)
        var sensorEventHumid = object: SensorEventListener{
            override fun onSensorChanged(sensorEvent: SensorEvent?) {
                val values = sensorEvent?.values
                if (values != null) {
                    //temperature.text  = values.get(0).toString()
                    var temp = values.get(0)
                    println(temp)
                    vlaznost=temp
                    writeHumidityToServer(vlaznost)
                }
            }
            override fun onAccuracyChanged(p0: Sensor?, p1: Int) {}
        }
        switch4.setOnCheckedChangeListener{_,isChecked->
            val updateSensorTemperature = object : Runnable {
                @RequiresApi(Build.VERSION_CODES.O)
                override fun run() {
                    runOnUiThread {
                        val cas = binding.cardTimeSensorDisplaySecondsInterval.text.toString().toInt()
                        println("Humid iz senozrja: $vlaznost")
                        writeHumidityToServer(vlaznost)
                        handler2.postDelayed(this, cas * 1000L)
                    }
                }
            }
            if (isChecked) {
                print("swithc2")
                sensorManagerTemperature.registerListener(sensorEventHumid, list2.get(0), SensorManager.SENSOR_DELAY_NORMAL)
                handler4.post(updateSensorTemperature)
                switch4.text ="Running"
            }
            else{
                print("swithc2")
                handler4.removeMessages(0);
                switch4.text ="Disabled"
            }
        }






        binding.btnDisplayBackGenerate.setOnClickListener {
            finish()
        }

        /*binding.buttonGenerate.setOnClickListener {
            if(isLocationSelected) {
                var listgeneratedNews = ListNovic()
                val loopCount = numberPicker.value
                val faker = Faker()
                val random = Random()
                for (i in 1..loopCount) {
                    val randomTitle = faker.book.title()
                    val currentDate = java.util.Date()

                    val firstName1 = faker.name.firstName()
                    val lastName1 = faker.name.lastName()
                    val firstName2 = faker.name.firstName()
                    val lastName2 = faker.name.lastName()
                    val fullName1 = "$firstName1 $lastName1"
                    val fullName2 = "$firstName2 $lastName2"
                    val authors = listOf(fullName1, fullName2)

                    val randomCategory1 = faker.book.genre()
                    val randomCategory2 = faker.book.genre()
                    val categories = listOf(randomCategory1, randomCategory2)

                    val locationX = textX.text.toString().toDouble()
                    val locationY = textY.text.toString().toDouble()

                    // Create a Location object
                    val location =
                        Location(type = "Point", coordinates = Pair(locationX, locationY))

                    val body: String = getString(R.string.loremipsun)
                    val randomId = random.nextInt(100).toString()
                    var novica1 = NewsArticle(
                        randomTitle,
                        "https://example1.com",
                        currentDate,
                        authors,
                        body,
                        categories,
                        location,
                        randomId,
                        1
                    )
                    // Your loop logic here
                    // This will be executed loopCount times
                    listgeneratedNews.addItem(novica1)
                }
                var novlistnovic = listgeneratedNews.getItems()
                finish()
            }
            else{
                Toast.makeText(this@GenerateDataActivity, "You did not select a location", Toast.LENGTH_SHORT).show()
            }
        }*/

    }
    private fun hasAmbientTemperatureSensor(): Boolean {
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        // Get the list of sensors
        val sensorList: List<Sensor> = sensorManager.getSensorList(Sensor.TYPE_AMBIENT_TEMPERATURE)

        // Check if the list is not empty, indicating the presence of the ambient temperature sensor
        return sensorList.isNotEmpty()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            REQUEST_CODE_LOCATION_PERMISSION -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                } else {
                    finish()
                }
            }
        }
    }

    private fun hasHumiditySensor(): Boolean {
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        // Get the list of sensors for TYPE_RELATIVE_HUMIDITY
        val sensorList: List<Sensor> = sensorManager.getSensorList(Sensor.TYPE_RELATIVE_HUMIDITY)

        // Check if the list is not empty, indicating the presence of the humidity sensor
        return sensorList.isNotEmpty()
    }

    companion object {
        private const val REQUEST_CODE_LOCATION_PERMISSION = 1001
        var realLastLocation: com.prvavaja.noviceprojekt.Location = Location("Point", Pair(0.0, 0.0))
    }

    private fun writeTempToServer(temp: Float) {
        myApplication.realm.writeBlocking {
            var article = NewsArticleRealm()
            article.authors = realmListOf("senzor")
            article.date = RealmInstant.now()
            article.categories = realmListOf("senzor-temperatura")
            article.content = temp.toString()
            article.title = "Temperatura iz senzorja"
            article.location = LocationRealm()
            article.location!!.type = realLastLocation.type
            println(realLastLocation.coordinates)
            article.location!!.coordinates = realmListOf(realLastLocation.coordinates.first, realLastLocation.coordinates.second)
            copyToRealm(article)
        }
    }

    private fun writeHumidityToServer(humidity: Float) {
        myApplication.realm.writeBlocking {
            var article = NewsArticleRealm()
            article.authors = realmListOf("senzor")
            article.date = RealmInstant.now()
            article.categories = realmListOf("senzor-vlažnost")
            article.content = humidity.toString()
            article.title = "Vlažnost iz senzorja"
            article.location = LocationRealm()
            article.location!!.type = realLastLocation.type
            println(realLastLocation.coordinates)
            article.location!!.coordinates = realmListOf(realLastLocation.coordinates.first, realLastLocation.coordinates.second)
            copyToRealm(article)
        }
    }
}