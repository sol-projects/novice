package com.prvavaja.noviceprojekt

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.prvavaja.noviceprojekt.databinding.ActivitySettingsBinding

class SettingsActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySettingsBinding
    private lateinit var myApplication:MyAplication
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_setings)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root) //ADD THIS LINE
        myApplication = application as MyAplication

        val numberPicker=binding.numberPickerSetings
        numberPicker.maxValue=40
        numberPicker.minValue=30
        numberPicker.value = 32

        binding.btnSetingsBack.setOnClickListener {
            finish()
        }
        binding.buttonSetingsActivate.setOnClickListener(){
            myApplication.maxtemperature=numberPicker.value
            myApplication.checkformaxTemp=true

        }
        binding.buttonSetingsDeactivate.setOnClickListener(){
            myApplication.maxtemperature=0
            myApplication.checkformaxTemp=false

        }
    }
}