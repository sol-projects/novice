package com.prvavaja.noviceprojekt

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.prvavaja.noviceprojekt.databinding.ActivityDisplayBinding
import com.prvavaja.noviceprojekt.databinding.ActivityMainBinding

class DisplayActivity : AppCompatActivity() {
    private lateinit var binding: ActivityDisplayBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDisplayBinding.inflate(layoutInflater)
        setContentView(binding.root)
        //setContentView(R.layout.activity_display)
        val titleTextView = binding.newsTitle
        val urlTextView = binding.textViewURL
        val dateTextView = binding.textViewDate
        val authorsTextView = binding.textViewDisplayAuthor
        val categoriesTextView = binding.textViewDisplayCatagories
        val locationTextView = binding.textViewDisplayLocation
        val contentTextView= binding.contentTextView


        val newsId = intent.getStringExtra("id")
        println(newsId)
        if (newsId != null) {
            titleTextView.setText(intent.getStringExtra("title"))
            urlTextView.setText(intent.getStringExtra("url"))
            dateTextView.setText(intent.getStringExtra("date"))
            authorsTextView.setText(intent.getStringExtra("authors"))
            categoriesTextView.setText(intent.getStringExtra("categories"))
            locationTextView.setText(intent.getStringExtra("cordinates"))
            contentTextView.setText(intent.getStringExtra("content"))
        }
        binding.btnDisplayBack.setOnClickListener {
            finish()
        }
    }
}