package com.prvavaja.noviceprojekt

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.prvavaja.noviceprojekt.databinding.ActivityGenarateBinding
import com.prvavaja.noviceprojekt.databinding.ActivityMainBinding
import org.osmdroid.config.Configuration
import android.content.Context
import android.widget.Toast
import io.github.serpro69.kfaker.Faker
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.api.IMapController
import org.osmdroid.events.MapEventsReceiver
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.MapEventsOverlay
import kotlinx.serialization.json.Json
import java.util.Random

class GenarateActivity : AppCompatActivity() {
    private lateinit var binding: ActivityGenarateBinding //ADD THIS LINE
    private lateinit var mapController: IMapController
    private var clickedLocation: GeoPoint? = null
    private var isLocationSelected: Boolean = false
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Configuration.getInstance().load(applicationContext,this.getPreferences(Context.MODE_PRIVATE))
        //setContentView(R.layout.activity_genarate)
        binding = ActivityGenarateBinding.inflate(layoutInflater) //ADD THIS LINE
        setContentView(binding.root)

        val textX=binding.textViewMapX
        val textY=binding.textViewMapY
        val numberPicker=binding.numberPickergenerate
        numberPicker.maxValue=10
        numberPicker.minValue=0
        numberPicker.value = 1
        val mapView = binding.map2
        mapView.setMultiTouchControls(true)
        mapController = mapView.controller

        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        //val sloveniaBoundingBox = BoundingBox(45.4214, 13.3754, 46.8762, 16.5645) // Adjust coordinates if needed
        //mapView.zoomToBoundingBox(sloveniaBoundingBox, true)
        val mapController = mapView.controller
        mapController.setZoom(8.7)
        val defaultLocation = GeoPoint(46.0767495796062, 14.853535025063707) // Adjust coordinates as needed
        mapController.setCenter(defaultLocation)

        val mapEventsOverlay = MapEventsOverlay(object : MapEventsReceiver {
            override fun singleTapConfirmedHelper(p: GeoPoint?): Boolean {
                // Handle the click event
                p?.let {
                    clickedLocation = p
                    println("Clicked Location: ${p.latitude}, ${p.longitude}")
                    textX.text=(p.latitude).toString()
                    textY.text=(p.longitude).toString()
                    isLocationSelected=true
                    // Save the location in your variable or perform any other actions
                }
                return true
            }

            override fun longPressHelper(p: GeoPoint?): Boolean {
                // Handle long press if needed
                return false
            }
        })
        mapView.overlays.add(0, mapEventsOverlay)

        binding.btnDisplayBackGenerate.setOnClickListener {
            finish()
        }
        binding.buttonGenerate.setOnClickListener {
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
                Toast.makeText(this@GenarateActivity, "You did not select a location", Toast.LENGTH_SHORT).show()
            }
        }

    }
}