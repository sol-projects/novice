package com.example.novinar

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.example.novinar.databinding.FragmentMapBinding
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.Marker
import com.google.android.gms.maps.model.MarkerOptions

class MapFragment : Fragment(), OnMapReadyCallback {

    private var _binding: FragmentMapBinding? = null
    private val binding get() = _binding!!
    private lateinit var googleMap: GoogleMap
    private val markerNewsMap = mutableMapOf<Marker, News>()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMapBinding.inflate(inflater, container, false)
        binding.mapView.onCreate(savedInstanceState)
        binding.mapView.getMapAsync(this)
        return binding.root
    }

    override fun onMapReady(map: GoogleMap) {
        googleMap = map
        loadMarkers()

        googleMap.setOnMarkerClickListener { marker ->
            marker.showInfoWindow()
            true
        }

        googleMap.setOnInfoWindowLongClickListener { marker ->
            markerNewsMap[marker]?.let { openDetailView(it) }
        }
    }

    private fun loadMarkers() {
        val newsList = NewsRepository.getNewsList() // Replace with your data source

        markerNewsMap.clear()
        googleMap.clear()

        for (news in newsList) {
            Log.d("MapFragment", "News: ${news.title}, Lat: ${news.latitude}, Long: ${news.longitude}")

            val latitude = news.latitude?.toDouble() ?: 15.5
            val longitude = news.longitude?.toDouble() ?: 0.0

            Log.d("MapFragment", "News: ${news.title}, Lat: ${news.latitude}, Long: ${news.longitude}")

            if (latitude == 0.0 && longitude == 0.0) continue

            val location = LatLng(latitude, longitude)
            val marker = googleMap.addMarker(
                MarkerOptions()
                    .position(location)
                    .title(news.title)
            )
            marker?.let {
                markerNewsMap[it] = news
                Log.d("MapFragment", "Added marker: ${news.title}")
            }
        }

        if (markerNewsMap.isNotEmpty()) {
            val firstMarker = markerNewsMap.keys.first()
            googleMap.moveCamera(CameraUpdateFactory.newLatLngZoom(firstMarker.position, 10f))
        } else {
            Log.d("MapFragment", "No markers to display")
        }
    }

    private fun openDetailView(news: News) {
        val intent = Intent(requireContext(), DetailViewFragment::class.java).apply {
            putExtra("news", news)
        }
        startActivity(intent)
    }

    override fun onResume() {
        super.onResume()
        binding.mapView.onResume()
    }

    override fun onPause() {
        super.onPause()
        binding.mapView.onPause()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        binding.mapView.onDestroy()
        _binding = null
    }
}
