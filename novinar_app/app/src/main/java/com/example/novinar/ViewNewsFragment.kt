package com.example.novinar

import android.content.Intent
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.novinar.api.ApiService
import okhttp3.MediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import org.json.JSONArray
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import java.io.IOException
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull

class ViewNewsFragment : Fragment() {

    private lateinit var apiService: ApiService
    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: NewsAdapter
    private lateinit var searchBar: EditText
    private lateinit var searchOptionSpinner: Spinner
    private lateinit var toggleSourceButton: Button
    private var fullNewsList: List<News> = emptyList()
    private val blockchainServerUrl = "http://192.168.1.119:5000"
    private var viewingBlockchainPosts = false
    private lateinit var captureImageButton: Button
    private val REQUEST_IMAGE_CAPTURE = 1
    private var photoFilePath: String? = null
    private val pythonServerUrl = "http://192.168.1.119:4000/compare"

    companion object {
        fun newInstance(apiService: ApiService): ViewNewsFragment {
            return ViewNewsFragment().apply {
                this.apiService = apiService
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_view_news, container, false)

        recyclerView = view.findViewById(R.id.newsRecyclerView)
        searchBar = view.findViewById(R.id.searchBar)
        searchOptionSpinner = view.findViewById(R.id.searchOptionSpinner)
        toggleSourceButton = view.findViewById(R.id.toggleSourceButton)

        recyclerView.layoutManager = LinearLayoutManager(requireContext())

        captureImageButton = view.findViewById(R.id.captureImageButton)
        captureImageButton.setOnClickListener {
            dispatchTakePictureIntent()
        }

        adapter = NewsAdapter(
            emptyList(),
            onDelete = { news -> deleteNews(news) },
            onLongClick = { news ->
                val fragment = DetailViewFragment.newInstance(news)
                parentFragmentManager.beginTransaction()
                    .replace(R.id.fragment_container, fragment)
                    .addToBackStack(null)
                    .commit()
            }
        )
        recyclerView.adapter = adapter

        setupSearchOptions()

        toggleSourceButton.text = "View Blockchain Posts"
        toggleSourceButton.setOnClickListener {
            viewingBlockchainPosts = !viewingBlockchainPosts
            if (viewingBlockchainPosts) {
                toggleSourceButton.text = "View All Posts"
                loadNewsFromBlockchain()
            } else {
                toggleSourceButton.text = "View Blockchain Posts"
                loadNewsFromDatabase()
            }
        }

        loadNewsFromDatabase()

        return view
    }

    private fun setupSearchOptions() {
        val searchOptions = listOf("Title", "Content")
        val adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, searchOptions)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        searchOptionSpinner.adapter = adapter

        searchBar.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val query = s.toString().trim()
                val selectedOption = searchOptionSpinner.selectedItem.toString()
                filterNews(query, selectedOption)
            }
        })
    }

    private fun loadNewsFromDatabase() {
        apiService.getNews().enqueue(object : Callback<List<News>> {
            override fun onResponse(call: Call<List<News>>, response: Response<List<News>>) {
                if (response.isSuccessful) {
                    val newsList = response.body()
                    if (!newsList.isNullOrEmpty()) {
                        fullNewsList = newsList
                        adapter.updateNews(newsList)
                    } else {
                        Toast.makeText(requireContext(), "No news found.", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    Log.e("API_ERROR", "Response Code: ${response.code()}, Error: ${response.errorBody()?.string()}")
                    Toast.makeText(requireContext(), "Failed to load news: ${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<List<News>>, t: Throwable) {
                Log.e("API_ERROR", "Error: ${t.message}", t)
                Toast.makeText(requireContext(), "Error: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun loadNewsFromBlockchain() {
        val client = OkHttpClient()
        val request = Request.Builder()
            .url("$blockchainServerUrl/chain")
            .get()
            .build()

        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: IOException) {
                Log.e("ViewNewsFragment", "Failed to fetch blockchain news: ${e.localizedMessage}")
                requireActivity().runOnUiThread {
                    Toast.makeText(requireContext(), "Failed to fetch blockchain news", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                if (response.isSuccessful) {
                    try {
                        val jsonResponse = JSONArray(response.body?.string() ?: "[]")
                        val blockchainNews = mutableListOf<News>()

                        for (i in 0 until jsonResponse.length()) {
                            val block = jsonResponse.getJSONObject(i)
                            val data = block.optJSONObject("data")
                            if (data != null) {
                                val news = News(
                                    _id = null,
                                    title = data.optString("title", "Untitled"),
                                    content = data.optString("content", ""),
                                    category = null,
                                    timestamp = block.optString("timestamp"),
                                    location = null,
                                    authors = null,
                                    categories = listOf(data.optString("categories", "Uncategorized")),
                                    views = null,
                                    url = null
                                )
                                blockchainNews.add(news)
                            } else {
                                Log.e("ViewNewsFragment", "Block $i missing data field")
                            }
                        }

                        requireActivity().runOnUiThread {
                            if (blockchainNews.isNotEmpty()) {
                                adapter.updateNews(blockchainNews)
                            } else {
                                Toast.makeText(requireContext(), "No blockchain news available", Toast.LENGTH_SHORT).show()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("ViewNewsFragment", "Error parsing blockchain data: ${e.localizedMessage}")
                        requireActivity().runOnUiThread {
                            Toast.makeText(requireContext(), "Error parsing blockchain data", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    Log.e("ViewNewsFragment", "Error fetching blockchain news. Code: ${response.code}")
                    requireActivity().runOnUiThread {
                        Toast.makeText(requireContext(), "Failed to fetch blockchain news: ${response.code}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }


    private fun filterNews(query: String, searchOption: String) {
        val filteredNews = if (query.isEmpty()) {
            fullNewsList
        } else {
            when (searchOption) {
                "Title" -> fullNewsList.filter { it.title.contains(query, ignoreCase = true) }
                "Content" -> fullNewsList.filter { it.content?.contains(query, ignoreCase = true) == true }
                else -> fullNewsList
            }
        }

        if (filteredNews.isEmpty()) {
            Toast.makeText(requireContext(), "No results found.", Toast.LENGTH_SHORT).show()
        }

        adapter.updateNews(filteredNews)
    }

    private fun deleteNews(news: News) {
        news._id?.let { newsId ->
            apiService.deleteNews(newsId).enqueue(object : Callback<Void> {
                override fun onResponse(call: Call<Void>, response: Response<Void>) {
                    if (response.isSuccessful) {
                        Toast.makeText(requireContext(), "News deleted successfully.", Toast.LENGTH_SHORT).show()
                        loadNewsFromDatabase()
                    } else {
                        Log.e("DeleteNews", "Failed to delete news. Code: ${response.code()}")
                        Toast.makeText(requireContext(), "Failed to delete news: ${response.code()}", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onFailure(call: Call<Void>, t: Throwable) {
                    Log.e("DeleteNews", "Error deleting news: ${t.message}")
                    Toast.makeText(requireContext(), "Error: ${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
        } ?: run {
            Toast.makeText(requireContext(), "Invalid news ID.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (takePictureIntent.resolveActivity(requireContext().packageManager) != null) {
            val photoFile = createImageFile()
            photoFile?.also {
                val photoURI = FileProvider.getUriForFile(
                    requireContext(),
                    "com.example.novinar.fileprovider",
                    it
                )
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }

    private fun createImageFile(): File? {
        val storageDir: File = requireContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES)!!
        return File.createTempFile("JPEG_${System.currentTimeMillis()}_", ".jpg", storageDir).apply {
            photoFilePath = absolutePath
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == AppCompatActivity.RESULT_OK) {
            photoFilePath?.let { path ->
                sendPhotoToServer(File(path))
            }
        }
    }

    private fun sendPhotoToServer(photoFile: File) {
        val client = OkHttpClient()
        val multipartBuilder = MultipartBody.Builder().setType(MultipartBody.FORM)

        multipartBuilder.addFormDataPart(
            "file",
            photoFile.name,
            RequestBody.create("image/jpeg".toMediaTypeOrNull(), photoFile)
        )

        val imageToNewsMap = mutableMapOf<String, News>()

        for (news in fullNewsList) {
            val imagePath = news.url
            if (imagePath != null) {
                val imageFile = File(imagePath)
                if (imageFile.exists()) {
                    val filename = imageFile.name
                    multipartBuilder.addFormDataPart(
                        "news_images",
                        filename,
                        RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
                    )
                    imageToNewsMap[filename] = news
                } else {
                    Log.w("ViewNewsFragment", "Image file does not exist: $imagePath")
                }
            } else {
                Log.w("ViewNewsFragment", "News item does not have an image URL: ${news.title}")
            }
        }

        val requestBody = multipartBuilder.build()
        val request = Request.Builder()
            .url(pythonServerUrl)
            .post(requestBody)
            .build()

        Log.d("ViewNewsFragment", "Sending photo and associated news images to server.")

        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: IOException) {
                Log.e("ViewNewsFragment", "Failed to send photo and news images: ${e.localizedMessage}")
                requireActivity().runOnUiThread {
                    Toast.makeText(requireContext(), "Failed to send photo", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                if (response.isSuccessful) {
                    val body = response.body
                    if (body != null) {
                        val responseBodyString = body.string()
                        Log.d("ViewNewsFragment", "Server response: $responseBodyString")

                        val jsonResponse = JSONArray(responseBodyString)
                        val orderedNews = mutableListOf<News>()

                        for (i in 0 until jsonResponse.length()) {
                            val item = jsonResponse.getJSONObject(i)
                            val filename = item.getString("filename")
                            val news = imageToNewsMap[filename]
                            if (news != null) {
                                orderedNews.add(news)
                            }
                        }

                        requireActivity().runOnUiThread {
                            adapter.updateNews(orderedNews)
                            Toast.makeText(requireContext(), "News reordered based on photo", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    Log.e("ViewNewsFragment", "Failed to reorder news: ${response.message}")
                    requireActivity().runOnUiThread {
                        Toast.makeText(requireContext(), "Failed to reorder news", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }
}
