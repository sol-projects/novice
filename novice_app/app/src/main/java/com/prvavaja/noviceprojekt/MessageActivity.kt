package com.prvavaja.noviceprojekt

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.prvavaja.noviceprojekt.databinding.ActivityMessageBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.osmdroid.api.IMapController
import org.osmdroid.config.Configuration
import org.osmdroid.events.MapEventsReceiver
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.overlay.MapEventsOverlay
import java.io.InputStream
import java.io.OutputStreamWriter
import java.net.URL
import java.nio.charset.StandardCharsets
import java.security.KeyManagementException
import java.security.KeyStore
import java.security.NoSuchAlgorithmException
import java.security.SecureRandom
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import java.util.Random
import java.util.concurrent.Executors
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.TrustManagerFactory
import javax.net.ssl.X509TrustManager

class MessageActivity : AppCompatActivity() {
    private lateinit var binding:  ActivityMessageBinding//ADD THIS LINE
    private lateinit var mapController: IMapController
    private var clickedLocation: GeoPoint? = null
    private var isLocationSelected: Boolean = false
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_masage)
        binding = ActivityMessageBinding.inflate(layoutInflater) //ADD THIS LINE
        setContentView(binding.root)

        Configuration.getInstance().load(applicationContext,this.getPreferences(Context.MODE_PRIVATE))
        //setContentView(R.layout.activity_genarate)

        val textX=binding.textViewMapXMessage
        val textY=binding.textViewMapYMessage
        val title=binding.editTextTitle
        val author=binding.editTextAuthor
        val content=binding.contentEditText


        val mapView = binding.map2
        mapView.setMultiTouchControls(true)
        mapController = mapView.controller

        val spiner=binding.editTextSpinerCatagory
        val adapter: ArrayAdapter<CharSequence> = ArrayAdapter.createFromResource(
            this,
            R.array.menu_items,
            android.R.layout.simple_spinner_item
        )

        // Specify the layout to use when the list of choices appears
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

        // Apply the adapter to the spinner
        spiner.adapter = adapter

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

        binding.btnDisplayBackMessage.setOnClickListener {
            finish()
        }
        binding.buttonMessage.setOnClickListener {
            val title_content=title.text.toString()
            val author_content=author.text.toString()
            val contnen_content=content.text.toString()

            if(isLocationSelected&&title_content.isNotEmpty()&&author_content.isNotEmpty()&&contnen_content.isNotEmpty()) {
                val random = Random()
                val currentDate = java.util.Date()
                val authors = listOf(author_content)
                val selectedItem = spiner.selectedItem.toString()
                val catagori_masage= "novice/$selectedItem"
                val categories = listOf(catagori_masage)
                val locationX = textX.text.toString().toDouble()
                val locationY = textY.text.toString().toDouble()
                val randomId = random.nextInt(100).toString()

                val location =
                    Location(type = "Point", coordinates = Pair(locationX, locationY))
                var message11 = NewsArticle(
                    title_content,
                    "",
                    currentDate,
                    authors,
                    contnen_content,
                    categories,
                    location,
                    randomId,
                    1
                )
                println(message11.toString())
                writeMessageToBlockchain(message11.toString())
                finish()
            }
            else{
                Toast.makeText(this@MessageActivity, "You did not select a location", Toast.LENGTH_SHORT).show()
            }
        }

    }

    private fun createOkHttpsClient(): OkHttpClient {
        val certificateStream: InputStream = this.assets.open("drawable/cert.pem")
        val certificate: X509Certificate = CertificateFactory.getInstance("X.509").generateCertificate(certificateStream) as X509Certificate

        val trustManagerFactory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
        val keyStore = KeyStore.getInstance(KeyStore.getDefaultType())

        keyStore.load(null, null)
        keyStore.setCertificateEntry("customCA", certificate)

        trustManagerFactory.init(keyStore)

        val trustManagers = trustManagerFactory.trustManagers

        val sslContext = SSLContext.getInstance("TLS")
        sslContext.init(null, trustManagers, null)

        return OkHttpClient.Builder()
            .sslSocketFactory(sslContext.socketFactory, trustManagers[0] as X509TrustManager)
            .hostnameVerifier { _, _ -> true }
            .build()
    }

    companion object {
        fun writeMessageToBlockchain(data: String) {
            val url = "http://10.0.2.2:49321/message"
            val client = OkHttpClient.Builder()
                .build()

            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val mediaType = "application/json".toMediaType()
                    val requestBody = data.toRequestBody(mediaType)

                    val request = Request.Builder()
                        .url(url)
                        .post(requestBody)
                        .build()

                    val response = client.newCall(request).execute()

                    if (response.isSuccessful) {
                        val responseBody = response.body?.string()
                        Log.d("blockchain", "Request successful. Response body: $responseBody")
                    } else {
                        Log.d("blockchain", "Request unsuccessful. Response code: ${response.code}")
                        val errorBody = response.body?.string()
                        Log.d("blockchain", "Error message: $errorBody")
                    }
                } catch (e: Exception) {
                    Log.e("blockchain", "Exception during request", e)
                }
            }
        }
    }
}