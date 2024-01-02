package com.prvavaja.noviceprojekt

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.prvavaja.noviceprojekt.databinding.ActivityCameraActivtyBinding
import com.prvavaja.noviceprojekt.databinding.ActivityMainBinding

class CameraActivty : AppCompatActivity() {
    private lateinit var binding: ActivityCameraActivtyBinding //ADD THIS LINE
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_camera_activty)
        binding = ActivityCameraActivtyBinding.inflate(layoutInflater) //ADD THIS LINE
        setContentView(binding.root)
        //setContentView(R.layout.activity_main)
        binding.btnCameraBack.setOnClickListener {  val intent = Intent(this, MainActivity::class.java)
            startActivity(intent) }
    }
}