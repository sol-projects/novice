package com.example.novinar

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class NewsAdapter(
    private var newsList: List<News>,
    private val onDelete: (News) -> Unit,
    private val onLongClick: (News) -> Unit
) : RecyclerView.Adapter<NewsAdapter.NewsViewHolder>() {

    class NewsViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleTextView: TextView = view.findViewById(R.id.textViewTitle)
        val contentTextView: TextView = view.findViewById(R.id.textViewContent)
        val deleteButton: Button = view.findViewById(R.id.buttonDelete)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NewsViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_news, parent, false)
        return NewsViewHolder(view)
    }

    override fun onBindViewHolder(holder: NewsViewHolder, position: Int) {
        val news = newsList[position]

        holder.titleTextView.text = news.title
        //holder.contentTextView.text = news.content


        // Delete button action
        holder.deleteButton.setOnClickListener { onDelete(news) }

        // Long click action
        holder.itemView.setOnLongClickListener {
            onLongClick(news)
            true
        }
    }

    override fun getItemCount(): Int = newsList.size

    fun updateNews(newNewsList: List<News>) {
        newsList = newNewsList
        notifyDataSetChanged()
    }
}

