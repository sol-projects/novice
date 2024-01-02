package com.prvavaja.noviceprojekt

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.prvavaja.noviceprojekt.databinding.ActivitySetingsBinding

class SetingsActivity : AppCompatActivity() {
    private lateinit var binding:ActivitySetingsBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_setings)
        binding = ActivitySetingsBinding.inflate(layoutInflater)
        setContentView(binding.root) //ADD THIS LINE

        binding.btnSetingsBack.setOnClickListener {  val intent = Intent(this, MainActivity::class.java)
            startActivity(intent) }
    }
}