package com.example.novinar

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.Spinner
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.novinar.api.ApiService
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class ViewNewsFragment : Fragment() {

    private lateinit var apiService: ApiService
    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: NewsAdapter
    private lateinit var searchBar: EditText
    private lateinit var searchOptionSpinner: Spinner
    private var fullNewsList: List<News> = emptyList()

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
        loadNewsFromDatabase()

        return view
    }

    private fun setupSearchOptions() {
        val searchOptions = listOf("Title", "PlanB")
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
                    Toast.makeText(requireContext(), "Failed to load news.", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<List<News>>, t: Throwable) {
                Toast.makeText(requireContext(), "Error: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun filterNews(query: String, searchOption: String) {
        val filteredNews = if (query.isEmpty()) {
            fullNewsList
        } else {
            when (searchOption) {
                "Title" -> fullNewsList.filter { it.title.contains(query, ignoreCase = true) }
                "PlanB" -> fullNewsList.filter { it.content?.contains(query, ignoreCase = true) == true } // tu mas za slike
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
                        Toast.makeText(requireContext(), "Failed to delete news.", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onFailure(call: Call<Void>, t: Throwable) {
                    Toast.makeText(requireContext(), "Error: ${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
        } ?: run {
            Toast.makeText(requireContext(), "Invalid news ID.", Toast.LENGTH_SHORT).show()
        }
    }
}
