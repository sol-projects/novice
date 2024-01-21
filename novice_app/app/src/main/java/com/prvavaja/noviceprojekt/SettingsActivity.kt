package com.prvavaja.noviceprojekt

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.prvavaja.noviceprojekt.databinding.ActivitySettingsBinding

class SettingsActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySettingsBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_setings)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root) //ADD THIS LINE

        val numberPicker=binding.numberPickerSetings
        numberPicker.maxValue=10
        numberPicker.minValue=0
        numberPicker.value = 1

        binding.btnSetingsBack.setOnClickListener {  val intent = Intent(this, MainActivity::class.java)
            startActivity(intent) }
    }
}