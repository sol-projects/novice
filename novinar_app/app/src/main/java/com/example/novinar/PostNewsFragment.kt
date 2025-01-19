package com.example.novinar

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.location.Location
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.novinar.api.ApiService
import com.google.android.gms.location.LocationServices
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import org.json.JSONArray
import org.json.JSONObject
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException

class PostNewsFragment : Fragment() {

    private lateinit var apiService: ApiService
    private lateinit var imageView: ImageView
    private var capturedImageBitmap: Bitmap? = null
    private val REQUEST_IMAGE_CAPTURE = 1
    private val REQUEST_PERMISSIONS = 100
    private var currentLatitude: Double = 0.0
    private var currentLongitude: Double = 0.0
    private val blockchainServerUrl = "http://192.168.1.119:5000"

    companion object {
        fun newInstance(apiService: ApiService): PostNewsFragment {
            val fragment = PostNewsFragment()
            fragment.apiService = apiService
            return fragment
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_post_news, container, false)

        val titleInput: EditText = view.findViewById(R.id.editTextTitle)
        val contentInput: EditText = view.findViewById(R.id.editTextContent)
        val categorySpinner: Spinner = view.findViewById(R.id.spinnerCategory)
        val postBlockchainButton: Button = view.findViewById(R.id.buttonPostBlockchain)
        val postButton: Button = view.findViewById(R.id.buttonPost)
        val captureButton: Button = view.findViewById(R.id.buttonCaptureImage)
        imageView = view.findViewById(R.id.imageViewCaptured)

        val categories = listOf("Politics", "Business", "Technology", "Sports", "Entertainment")
        val adapter =
            ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, categories)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        categorySpinner.adapter = adapter

        captureButton.setOnClickListener {
            checkAndRequestPermissions()
        }

        postButton.setOnClickListener {
            val title = titleInput.text.toString()
            val content = contentInput.text.toString()
            val category = categorySpinner.selectedItem.toString()

            if (title.isNotBlank() && content.isNotBlank()) {
                val imageFile = capturedImageBitmap?.let { bitmapToFile(it) }
                postNewsToServer(title, content, listOf(category), imageFile)
            } else {
                Toast.makeText(
                    requireContext(),
                    "Title and content cannot be empty!",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

        postBlockchainButton.setOnClickListener {
            val title = titleInput.text.toString()
            val content = contentInput.text.toString()
            val category = categorySpinner.selectedItem.toString()

            if (title.isNotBlank() && content.isNotBlank()) {
                postToBlockchain(title, content, listOf(category))
            } else {
                Toast.makeText(
                    requireContext(),
                    "Title and content cannot be empty!",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }


        fetchUserLocation()

        return view
    }

    private fun fetchUserLocation() {
        if (ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.ACCESS_FINE_LOCATION
            )
            == PackageManager.PERMISSION_GRANTED
        ) {
            val fusedLocationClient =
                LocationServices.getFusedLocationProviderClient(requireContext())
            fusedLocationClient.lastLocation.addOnSuccessListener { location: Location? ->
                if (location != null) {
                    currentLatitude = location.latitude
                    currentLongitude = location.longitude
                    Log.d(
                        "PostNewsFragment",
                        "Location fetched: $currentLatitude, $currentLongitude"
                    )
                } else {
                    Toast.makeText(requireContext(), "Unable to fetch location", Toast.LENGTH_SHORT)
                        .show()
                }
            }
        } else {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                REQUEST_PERMISSIONS
            )
        }
    }

    private fun postToBlockchain(title: String, content: String, categories: List<String>) {
        val jsonObject = JSONObject().apply {
            put("title", title)
            put("content", content)
            put("categories", JSONArray(categories))
            put("location", JSONObject().apply {
                put("type", "Point")
                put("coordinates", JSONArray().apply {
                    put(currentLongitude)
                    put(currentLatitude)
                })
            })
        }

        val requestBody = RequestBody.create(
            "application/json".toMediaTypeOrNull(),
            jsonObject.toString()
        )

        val client = OkHttpClient()
        val request = Request.Builder()
            .url("$blockchainServerUrl/add_block")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: IOException) {
                Log.e("PostNewsFragment", "Error posting to blockchain: ${e.localizedMessage}")
                requireActivity().runOnUiThread {
                    Toast.makeText(
                        requireContext(),
                        "Error: ${e.localizedMessage}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                requireActivity().runOnUiThread {
                    if (response.isSuccessful) {
                        Toast.makeText(
                            requireContext(),
                            "Posted to Blockchain successfully!",
                            Toast.LENGTH_SHORT
                        ).show()
                    } else {
                        Log.e(
                            "PostNewsFragment",
                            "Failed to post to blockchain. Code: ${response.code}"
                        )
                        Toast.makeText(
                            requireContext(),
                            "Failed to post to blockchain: ${response.code}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        })
    }

    private fun checkAndRequestPermissions() {
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_PERMISSIONS
            )
        } else {
            dispatchTakePictureIntent()
        }
    }

    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (takePictureIntent.resolveActivity(requireActivity().packageManager) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
        } else {
            Toast.makeText(requireContext(), "No Camera app found!", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            val bitmap = data?.extras?.get("data") as? Bitmap
            if (bitmap != null) {
                capturedImageBitmap = bitmap
                imageView.setImageBitmap(bitmap)
            } else {
                Toast.makeText(requireContext(), "Failed to capture image!", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    private fun bitmapToFile(bitmap: Bitmap): File? {
        return try {
            val file = File(requireContext().cacheDir, "captured_image.jpg")
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
            file.outputStream().use { it.write(outputStream.toByteArray()) }
            file
        } catch (e: Exception) {
            Log.e("PostNewsFragment", "Error saving bitmap to file: ${e.localizedMessage}")
            null
        }
    }

    private fun postNewsToServer(
        title: String,
        content: String,
        categories: List<String>,
        imageFile: File?
    ) {
        val jsonObject = JSONObject().apply {
            put("title", title)
            put("content", content)
            put("categories", JSONArray(categories))
            put("location", JSONObject().apply {
                put("type", "Point")
                put("coordinates", JSONArray().apply {
                    put(currentLongitude)
                    put(currentLatitude)
                })
            })
        }

        val requestBody = RequestBody.create(
            "application/json".toMediaTypeOrNull(),
            jsonObject.toString()
        )

        apiService.postNews(requestBody).enqueue(object : Callback<Void> {
            override fun onResponse(call: Call<Void>, response: Response<Void>) {
                if (response.isSuccessful) {
                    Toast.makeText(
                        requireContext(),
                        "News posted successfully!",
                        Toast.LENGTH_SHORT
                    ).show()
                } else {
                    Log.e("PostNewsFragment", "Failed to post news. Code: ${response.code()}")
                    Toast.makeText(
                        requireContext(),
                        "Failed to post news: ${response.code()}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

            override fun onFailure(call: Call<Void>, t: Throwable) {
                Log.e("PostNewsFragment", "Error posting news: ${t.localizedMessage}")
                Toast.makeText(requireContext(), "Error: ${t.localizedMessage}", Toast.LENGTH_SHORT)
                    .show()
            }
        })
    }
}
