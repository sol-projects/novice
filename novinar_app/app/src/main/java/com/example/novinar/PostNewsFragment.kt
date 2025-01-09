package com.example.novinar

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.example.novinar.api.ApiService
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class PostNewsFragment : Fragment() {
    private lateinit var apiService: ApiService
    private var isEditing: Boolean = false
    private var editingNewsId: String? = null

    companion object {
        fun newInstance(apiService: ApiService): PostNewsFragment {
            val fragment = PostNewsFragment()
            fragment.apiService = apiService // Set the API service instance
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
        val postButton: Button = view.findViewById(R.id.buttonPost)

        // Check if the fragment is in editing mode
        arguments?.let {
            isEditing = it.getBoolean("isEditing", false)
            editingNewsId = it.getString("newsId", null)
            titleInput.setText(it.getString("title", ""))
            contentInput.setText(it.getString("content", ""))
        }

        postButton.text = if (isEditing) "Update News" else "Post News"

        postButton.setOnClickListener {
            val title = titleInput.text.toString()
            val content = contentInput.text.toString()

            if (title.isNotEmpty() && content.isNotEmpty()) {
                if (isEditing) {
                    updateNews(editingNewsId, title, content)
                } else {
                    postNews(title, content)
                }
            } else {
                Toast.makeText(context, "Please fill out all fields", Toast.LENGTH_SHORT).show()
            }
        }

        return view
    }

    private fun postNews(title: String, content: String) {
        val news = News(
            _id = null,
            title = title,
            content = content,
            latitude = 0.0,
            longitude = 0.0,
            timestamp = "null"
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

    private fun updateNews(id: String?, title: String, content: String) {
        if (id == null) {
            Toast.makeText(context, "Invalid news ID", Toast.LENGTH_SHORT).show()
            return
        }

        val updatedNews = News(
            _id = id,
            title = title,
            content = content,
            latitude = 0.0,
            longitude = 0.0,
            timestamp = "null"
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

