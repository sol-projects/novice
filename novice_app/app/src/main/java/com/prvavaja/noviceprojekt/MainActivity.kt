package com.prvavaja.noviceprojekt

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.prvavaja.noviceprojekt.databinding.ActivityMainBinding
import java.text.SimpleDateFormat

class MainActivity : AppCompatActivity(), MyAdapter.OnItemClickListener {
    private lateinit var binding: ActivityMainBinding //ADD THIS LINE
    private lateinit var myApplication:MyAplication
    private lateinit var adapter: MyAdapter
            override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater) //ADD THIS LINE
        setContentView(binding.root)
        myApplication = application as MyAplication
        val recyclerView: RecyclerView = binding.recyclerView
        //setContentView(R.layout.activity_main)

        //TEST DA VIDIM CE SO OBJEKTI NOVIC PRAVILNO SJRANJENI V LISTU
        var list1=myApplication.list
        var list2=list1.getItems()
        //ADAPTER
        adapter = MyAdapter(myApplication.list.getItems())
        adapter.setOnItemClickListener(this)
        recyclerView.adapter = adapter
        recyclerView.layoutManager = LinearLayoutManager(this) // Use GridLayoutManager for a grid layout

        binding.btnSetings.setOnClickListener {  val intent = Intent(this, SetingsActivity::class.java)
            startActivity(intent) }
        binding.buttonCamera.setOnClickListener {  val intent = Intent(this, CameraActivty::class.java)
            startActivity(intent) }
    }
    override fun onItemClick(novica: Novica) {
        val intent = Intent(this, DisplayActivity::class.java)

        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        val formattedDate = dateFormat.format(novica.date)
        val formattedAuthors = novica.authors.joinToString("\", \"", "[\"", "\"]")
        val formattedCategories = novica.categories.joinToString("\", \"", "[\"", "\"]")
        val coordinates = "[${novica.location.coordinates.first}, ${novica.location.coordinates.second}]"
        // Add extra data to the intent to pass the UUID of the clicked Sadje object
        intent.putExtra("id", novica._id)
        intent.putExtra("title",novica.title)
        intent.putExtra("url", novica.url)
        intent.putExtra("date", formattedDate)
        intent.putExtra("authors",formattedAuthors)
        intent.putExtra("categories",formattedCategories)
        intent.putExtra("cordinates",coordinates)
        intent.putExtra("content",novica.content)
        intent.putExtra("__v",novica.__v)
        // Start the MainActivity using the created intent
        //startActivity(intent)
        startActivity(intent)
    }
}