package com.prvavaja.noviceprojekt

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat

class MyAdapter (private var newsList: List<Novica>) : RecyclerView.Adapter<NewsViewHolder>()  {
    private var onItemClickListener: OnItemClickListener? = null
    private var onItemLongClickListener: OnItemLongClickListener? = null
    interface OnItemLongClickListener {
        fun onItemLongClick(stadion: Novica): Boolean
    }
    fun setOnItemLongClickListener(listener: OnItemLongClickListener) {
        this.onItemLongClickListener = listener
    }
    interface OnItemClickListener {
        fun onItemClick(stadion: Novica)
    }
    fun setOnItemClickListener(listener: OnItemClickListener) {
        this.onItemClickListener = listener
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NewsViewHolder {
        val itemView = LayoutInflater.from(parent.context).inflate(R.layout.list_item, parent, false)
        return NewsViewHolder(itemView)
    }
    override fun onBindViewHolder(holder: NewsViewHolder, position: Int) {
        val news = newsList[position]
        holder.textViewId.text = "ID: ${news._id}"
        holder.textViewTitle.text ="TITLE: ${news.title}"
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        val formattedDate = dateFormat.format(news.date)
        holder.textViewDate.text ="FATE: ${formattedDate}"
        val formattedCategories = news.categories.joinToString("\", \"", "[\"", "\"]")
        val coordinates = "[${news.location.coordinates.first}, ${news.location.coordinates.second}]"
        val formattedAuthors = news.authors.joinToString("\", \"", "[\"", "\"]")
        holder.textViewCatagories.text ="CATEGORIES: ${formattedCategories}"
        holder.textViewAuthors.text ="AUTHORS: ${formattedAuthors}"
        holder.textViewLocations.text ="LOCATION: ${coordinates}"

        // Set click listener
        holder.itemView.setOnClickListener {
            onItemClickListener?.onItemClick(news)
        }
        holder.itemView.setOnLongClickListener {
            onItemLongClickListener?.onItemLongClick(news)
            true // Consume the long click event
        }
        // Bind other data to TextViews
    }

    override fun getItemCount(): Int {
        return newsList.size
    }
    //TO KLICES DA UPDATAS IZGLED RECYCLERVIEWA
    fun updateData(newData: List<Novica>) {
        newsList = newData
        notifyDataSetChanged()
    }
}