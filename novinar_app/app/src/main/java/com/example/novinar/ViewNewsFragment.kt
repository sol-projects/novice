package com.example.novinar

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
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

    companion object {
        fun newInstance(apiService: ApiService): ViewNewsFragment {
            val fragment = ViewNewsFragment()
            fragment.apiService = apiService
            return fragment
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_view_news, container, false)

        recyclerView = view.findViewById(R.id.newsRecyclerView)
        recyclerView.layoutManager = LinearLayoutManager(requireContext())

        adapter = NewsAdapter(emptyList(),
            onEdit = { news ->
                // Handle edit action
                Toast.makeText(requireContext(), "Edit: ${news.title}", Toast.LENGTH_SHORT).show()
            },
            onDelete = { news ->
                // Handle delete action
                Toast.makeText(requireContext(), "Delete: ${news.title}", Toast.LENGTH_SHORT).show()
            }
        )
        recyclerView.adapter = adapter

        loadNewsFromDatabase()

        return view
    }

    private fun loadNewsFromDatabase() {
        apiService.getNews().enqueue(object : Callback<List<News>> {
            override fun onResponse(call: Call<List<News>>, response: Response<List<News>>) {
                if (response.isSuccessful) {
                    val newsList = response.body()
                    if (!newsList.isNullOrEmpty()) {
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
}
