package com.prvavaja.noviceprojekt

import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class NewsViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
    val textViewId: TextView = itemView.findViewById(R.id.textViewIdZ)
    val textViewTitle: TextView = itemView.findViewById(R.id.textViewTitleZ)
    val textViewDate: TextView = itemView.findViewById(R.id.textViewDateZ)
    val textViewCatagories: TextView = itemView.findViewById(R.id.textViewCategoriesZ)
    val textViewAuthors: TextView = itemView.findViewById(R.id.textViewAuthorsZ)
    val textViewLocations: TextView = itemView.findViewById(R.id.textViewLocationZ)
    val imageViewStadion: ImageView =itemView.findViewById(R.id.imageView)
}