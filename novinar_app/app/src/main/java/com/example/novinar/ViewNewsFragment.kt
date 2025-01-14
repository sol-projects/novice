package com.example.novinar

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.ContentProviderCompat
import androidx.core.content.ContentProviderCompat.requireContext
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.novinar.api.ApiService
import com.example.novinar.api.RetrofitClient
import com.example.novinar.api.RetrofitClient.apiService
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response


class ViewNewsFragment : Fragment() {
    private lateinit var apiService: ApiService
    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: NewsAdapter

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
        recyclerView.layoutManager = LinearLayoutManager(requireContext())

        adapter = NewsAdapter(
            emptyList(),
            onEdit = { news ->
                // Navigate to PostNewsFragment for editing
                val fragment = PostNewsFragment.newInstance(apiService).apply {
                    arguments = Bundle().apply {
                        putBoolean("isEditing", true)
                        putString("newsId", news._id)
                        putString("title", news.title)
                        putString("content", news.content)
                    }
                }
                parentFragmentManager.beginTransaction()
                    .replace(R.id.fragment_container, fragment)
                    .addToBackStack(null)
                    .commit()
            },
            onDelete = { news ->
                // Show confirmation dialog for deleting news
                deleteNews(news)
            },
            onLongClick = { news ->
                // Navigate to DetailViewFragment on long click
                val fragment = DetailViewFragment.newInstance(news)
                parentFragmentManager.beginTransaction()
                    .replace(R.id.fragment_container, fragment)
                    .addToBackStack(null)
                    .commit()
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

    private fun deleteNews(news: News) {
        news._id?.let { newsId ->
            apiService.deleteNews(newsId).enqueue(object : Callback<Void> {
                override fun onResponse(call: Call<Void>, response: Response<Void>) {
                    if (response.isSuccessful) {
                        Toast.makeText(requireContext(), "News deleted successfully.", Toast.LENGTH_SHORT).show()
                        loadNewsFromDatabase() // Refresh the list
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
