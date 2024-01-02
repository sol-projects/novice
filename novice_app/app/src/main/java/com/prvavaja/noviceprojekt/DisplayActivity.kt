package com.prvavaja.noviceprojekt

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.prvavaja.noviceprojekt.databinding.ActivityDisplayBinding
import com.prvavaja.noviceprojekt.databinding.ActivityMainBinding

class DisplayActivity : AppCompatActivity() {
    private lateinit var binding: ActivityDisplayBinding //ADD THIS LINE
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
//DOBIM PODATKE OD IZBRANE NOVICE IN JIH DISPLEJAM
        binding = ActivityDisplayBinding.inflate(layoutInflater) //ADD THIS LINE
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
            //val clickedSadje =findSadjeByUuid(sadjeUuid) // Retrieve Sadje object using the UUID
            titleTextView.setText(intent.getStringExtra("title"))
            urlTextView.setText(intent.getStringExtra("url"))
            dateTextView.setText(intent.getStringExtra("date"))
            authorsTextView.setText(intent.getStringExtra("authors"))
            categoriesTextView.setText(intent.getStringExtra("categories"))
            locationTextView.setText(intent.getStringExtra("cordinates"))
            contentTextView.setText(intent.getStringExtra("content"))

            //populateEditTextFields(clickedSadje)
        }
        binding.btnDisplayBack.setOnClickListener {
            finish()
        }
    }
}