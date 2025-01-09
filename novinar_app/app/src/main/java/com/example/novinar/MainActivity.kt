package com.example.novinar

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.example.novinar.api.ApiService
import com.example.novinar.api.RetrofitClient
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.example.novinar.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private val apiService = RetrofitClient.apiService // Initialize the API service

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val navView = binding.navView

        // Load default fragment (ViewNewsFragment) on app startup
        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, ViewNewsFragment.newInstance(apiService))
                .commit()
        }

        // Handle bottom navigation item selection
        navView.setOnItemSelectedListener { item ->
            val selectedFragment: Fragment = when (item.itemId) {
                R.id.navigation_post -> PostNewsFragment.newInstance(apiService)
                R.id.navigation_view -> ViewNewsFragment.newInstance(apiService)
                R.id.navigation_map -> MapFragment()
                R.id.navigation_sensor -> SensorDataFragment()
                else -> ViewNewsFragment.newInstance(apiService)
            }
            supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, selectedFragment)
                .commit()
            true
        }
    }
}

