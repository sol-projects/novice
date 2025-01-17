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
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.ByteArrayOutputStream
import java.io.File

class PostNewsFragment : Fragment() {

    private lateinit var apiService: ApiService
    private lateinit var imageView: ImageView
    private lateinit var captureButton: Button
    private lateinit var postButton: Button
    private lateinit var removeImageButton: Button
    private var capturedImageBitmap: Bitmap? = null
    private val REQUEST_IMAGE_CAPTURE = 1
    private val REQUEST_PERMISSIONS = 100
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var currentLatitude: Double = 0.0
    private var currentLongitude: Double = 0.0

    companion object {
        fun newInstance(apiService: ApiService): PostNewsFragment {
            return PostNewsFragment().apply {
                this.apiService = apiService
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_post_news, container, false)

        imageView = view.findViewById(R.id.imageViewCaptured)
        captureButton = view.findViewById(R.id.buttonCaptureImage)
        removeImageButton = view.findViewById(R.id.buttonRemoveImage)
        postButton = view.findViewById(R.id.buttonPost)

        val titleInput: EditText = view.findViewById(R.id.editTextTitle)
        val contentInput: EditText = view.findViewById(R.id.editTextContent)
        val categorySpinner: Spinner = view.findViewById(R.id.spinnerCategory)

        // Setup categories
        val categories = listOf("Politics", "Business", "Technology", "Sports", "Entertainment")
        val adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, categories)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        categorySpinner.adapter = adapter

        // Capture image button
        captureButton.setOnClickListener {
            checkAndRequestPermissions()
        }

        // Remove image button
        removeImageButton.setOnClickListener {
            capturedImageBitmap = null
            imageView.setImageResource(R.drawable.placeholder_image) // Reset to placeholder
        }

        // Post news button
        postButton.setOnClickListener {
            val title = titleInput.text.toString()
            val content = contentInput.text.toString()
            val category = categorySpinner.selectedItem.toString()

            if (title.isNotBlank() && content.isNotBlank()) {
                val imageFile = capturedImageBitmap?.let { bitmapToFile(it) }
                postNewsToServer(title, content, category, imageFile)
            } else {
                Toast.makeText(requireContext(), "Title and content cannot be empty!", Toast.LENGTH_SHORT).show()
            }
        }

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(requireContext())
        fetchUserLocation()

        return view
    }
    private fun fetchUserLocation() {
        if (ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            fusedLocationClient.lastLocation.addOnSuccessListener { location: Location? ->
                if (location != null) {
                    currentLatitude = location.latitude
                    currentLongitude = location.longitude
                } else {
                    Toast.makeText(requireContext(), "Unable to fetch location", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun checkAndRequestPermissions() {
        if (ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
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

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                dispatchTakePictureIntent()
            } else {
                Toast.makeText(requireContext(), "Camera permission is required!", Toast.LENGTH_SHORT).show()
            }
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
                Toast.makeText(requireContext(), "Failed to capture image!", Toast.LENGTH_SHORT).show()
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


    private fun postNewsToServer(title: String, content: String, category: String, imageFile: File?) {
        val titlePart = RequestBody.create("text/plain".toMediaTypeOrNull(), title)
        val contentPart = RequestBody.create("text/plain".toMediaTypeOrNull(), content)
        val categoryPart = RequestBody.create("text/plain".toMediaTypeOrNull(), category)
        val latitudePart = RequestBody.create("text/plain".toMediaTypeOrNull(), currentLatitude.toString())
        val longitudePart = RequestBody.create("text/plain".toMediaTypeOrNull(), currentLongitude.toString())

        val imagePart = if (imageFile != null) {
            MultipartBody.Part.createFormData(
                "image", imageFile.name, RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
            )
        } else {
            MultipartBody.Part.createFormData(
                "image", "placeholder.txt", RequestBody.create("text/plain".toMediaTypeOrNull(), "no_image")
            )
        }

        val call = apiService.addNews(titlePart, contentPart, categoryPart, latitudePart, longitudePart, imagePart)

        call.enqueue(object : Callback<Void> {
            override fun onResponse(call: Call<Void>, response: Response<Void>) {
                if (response.isSuccessful) {
                    Toast.makeText(requireContext(), "News posted successfully!", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(requireContext(), "Failed to post news!", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Void>, t: Throwable) {
                Toast.makeText(requireContext(), "Error: ${t.localizedMessage}", Toast.LENGTH_SHORT).show()
            }
        })
    }
}
