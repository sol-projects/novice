package com.example.novinar

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.novinar.api.ApiService
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.IOException

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
        val adapter =
            ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, searchOptions)
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
                        Toast.makeText(requireContext(), "No news found.", Toast.LENGTH_SHORT)
                            .show()
                    }
                } else {
                    Log.e(
                        "API_ERROR",
                        "Response Code: ${response.code()}, Error: ${
                            response.errorBody()?.string()
                        }"
                    )
                    Toast.makeText(
                        requireContext(),
                        "Failed to load news: ${response.code()}",
                        Toast.LENGTH_SHORT
                    ).show()
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
                    Toast.makeText(
                        requireContext(),
                        "Failed to fetch blockchain news",
                        Toast.LENGTH_SHORT
                    ).show()
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
                                    categories = listOf(
                                        data.optString(
                                            "categories",
                                            "Uncategorized"
                                        )
                                    ),
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
                                Toast.makeText(
                                    requireContext(),
                                    "No blockchain news available",
                                    Toast.LENGTH_SHORT
                                ).show()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(
                            "ViewNewsFragment",
                            "Error parsing blockchain data: ${e.localizedMessage}"
                        )
                        requireActivity().runOnUiThread {
                            Toast.makeText(
                                requireContext(),
                                "Error parsing blockchain data",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                } else {
                    Log.e(
                        "ViewNewsFragment",
                        "Error fetching blockchain news. Code: ${response.code}"
                    )
                    requireActivity().runOnUiThread {
                        Toast.makeText(
                            requireContext(),
                            "Failed to fetch blockchain news: ${response.code}",
                            Toast.LENGTH_SHORT
                        ).show()
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
                "Content" -> fullNewsList.filter {
                    it.content?.contains(
                        query,
                        ignoreCase = true
                    ) == true
                }

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
                        Toast.makeText(
                            requireContext(),
                            "News deleted successfully.",
                            Toast.LENGTH_SHORT
                        ).show()
                        loadNewsFromDatabase()
                    } else {
                        Log.e("DeleteNews", "Failed to delete news. Code: ${response.code()}")
                        Toast.makeText(
                            requireContext(),
                            "Failed to delete news: ${response.code()}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }

                override fun onFailure(call: Call<Void>, t: Throwable) {
                    Log.e("DeleteNews", "Error deleting news: ${t.message}")
                    Toast.makeText(requireContext(), "Error: ${t.message}", Toast.LENGTH_SHORT)
                        .show()
                }
            })
        } ?: run {
            Toast.makeText(requireContext(), "Invalid news ID.", Toast.LENGTH_SHORT).show()
        }
    }
}
