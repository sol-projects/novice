package com.example.novinar

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import com.example.novinar.api.ApiService
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.ByteArrayOutputStream

class PostNewsFragment : Fragment() {

    private lateinit var apiService: ApiService
    private var isEditing: Boolean = false
    private var editingNewsId: String? = null
    private var capturedImage: Bitmap? = null

    companion object {
        private const val REQUEST_IMAGE_CAPTURE = 1

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
        if (!::apiService.isInitialized) {
            throw IllegalStateException("apiService is not initialized. Use newInstance to create the fragment.")
        }

        val view = inflater.inflate(R.layout.fragment_post_news, container, false)

        val titleInput: EditText = view.findViewById(R.id.editTextTitle)
        val contentInput: EditText = view.findViewById(R.id.editTextContent)
        val categorySpinner: Spinner = view.findViewById(R.id.spinnerCategory)
        val cameraButton: Button = view.findViewById(R.id.buttonCaptureImage)
        val postButton: Button = view.findViewById(R.id.buttonPost)
        val imageView: ImageView = view.findViewById(R.id.imageViewCaptured)

        // Predefined categories
        val categories = listOf(
            "Politics", "Business", "Technology", "Sports", "Entertainment",
            "Health", "Science", "World", "Lifestyle", "Environment"
        )
        val adapter = ArrayAdapter(
            requireContext(),
            android.R.layout.simple_spinner_item,
            categories
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        categorySpinner.adapter = adapter

        // Check if the fragment is in editing mode
        arguments?.let {
            isEditing = it.getBoolean("isEditing", false)
            editingNewsId = it.getString("newsId", null)
            titleInput.setText(it.getString("title", ""))
            contentInput.setText(it.getString("content", ""))
        }

        postButton.text = if (isEditing) "Update News" else "Post News"

        // Capture image on button click
        cameraButton.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (takePictureIntent.resolveActivity(requireActivity().packageManager) != null) {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }

        // Handle post button click
        postButton.setOnClickListener {
            val title = titleInput.text.toString().trim()
            val content = contentInput.text.toString().trim()
            val category = categorySpinner.selectedItem.toString()

            if (title.isEmpty() || content.isEmpty() || capturedImage == null) {
                Toast.makeText(context, "Please fill out all fields and capture an image", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val encodedImage = encodeImageToBase64(capturedImage!!)
            if (isEditing) {
                updateNews(editingNewsId, title, content, category, encodedImage)
            } else {
                postNews(title, content, category, encodedImage)
            }
        }

        return view
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            val imageBitmap = data?.extras?.get("data") as Bitmap
            capturedImage = imageBitmap
            view?.findViewById<ImageView>(R.id.imageViewCaptured)?.apply {
                visibility = View.VISIBLE
                setImageBitmap(imageBitmap)
            }
        }
    }

    private fun encodeImageToBase64(bitmap: Bitmap): String {
        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream)
        val byteArray = byteArrayOutputStream.toByteArray()
        return Base64.encodeToString(byteArray, Base64.DEFAULT)
    }

    private fun postNews(title: String, content: String, category: String, image: String) {
        val news = News(
            _id = null,
            title = title,
            content = content,
            category = category,
            image = image,
            latitude = 0.0,
            longitude = 0.0,
            timestamp = null
        )

        apiService.addNews(news).enqueue(object : Callback<Void> {
            override fun onResponse(call: Call<Void>, response: Response<Void>) {
                if (response.isSuccessful) {
                    Toast.makeText(context, "News posted successfully!", Toast.LENGTH_SHORT).show()
                    requireActivity().supportFragmentManager.popBackStack()
                } else {
                    Toast.makeText(context, "Failed to post news", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Void>, t: Throwable) {
                Toast.makeText(context, "Error: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun updateNews(id: String?, title: String, content: String, category: String, image: String) {
        if (id == null) {
            Toast.makeText(context, "Invalid news ID", Toast.LENGTH_SHORT).show()
            return
        }

        val updatedNews = News(
            _id = id,
            title = title,
            content = content,
            category = category,
            image = image,
            latitude = 0.0,
            longitude = 0.0,
            timestamp = null
        )

        apiService.editNews(id, updatedNews).enqueue(object : Callback<Void> {
            override fun onResponse(call: Call<Void>, response: Response<Void>) {
                if (response.isSuccessful) {
                    Toast.makeText(context, "News updated successfully!", Toast.LENGTH_SHORT).show()
                    requireActivity().supportFragmentManager.popBackStack()
                } else {
                    Toast.makeText(context, "Failed to update news", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Void>, t: Throwable) {
                Toast.makeText(context, "Error: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }
}
