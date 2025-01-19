package com.example.novinar

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.example.novinar.api.ApiService
import com.example.novinar.api.RetrofitClient
import com.example.novinar.databinding.FragmentMapBinding
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.Marker
import com.google.android.gms.maps.model.MarkerOptions
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response


class MapFragment : Fragment(), OnMapReadyCallback {

    private var _binding: FragmentMapBinding? = null
    private val binding get() = _binding!!
    private lateinit var googleMap: GoogleMap
    private val markerNewsMap = mutableMapOf<Marker, News>()
    private val apiService: ApiService by lazy {
        RetrofitClient.apiService
    }

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
        Log.d("MapFragment", "Google Map is ready")
        fetchNewsAndLoadMarkers()

        googleMap.setOnMarkerClickListener { marker ->
            marker.showInfoWindow()
            true
        }

        googleMap.setOnInfoWindowLongClickListener { marker ->
            markerNewsMap[marker]?.let { openDetailView(it) }
        }
    }

    private fun fetchNewsAndLoadMarkers() {
        apiService.getNews().enqueue(object : Callback<List<News>> {
            override fun onResponse(call: Call<List<News>>, response: Response<List<News>>) {
                if (response.isSuccessful) {
                    val newsList = response.body()
                    Log.d("MapFragment", "API Response: $newsList")

                    if (!newsList.isNullOrEmpty()) {
                        loadMarkers(newsList)
                    } else {
                        Toast.makeText(requireContext(), "No news found.", Toast.LENGTH_SHORT)
                            .show()
                    }
                } else {
                    Log.e(
                        "MapFragment",
                        "Response failed. Code: ${response.code()}, Error: ${
                            response.errorBody()?.string()
                        }"
                    )
                    Toast.makeText(requireContext(), "Failed to load news.", Toast.LENGTH_SHORT)
                        .show()
                }
            }

            override fun onFailure(call: Call<List<News>>, t: Throwable) {
                Log.e("MapFragment", "API call failed. Error: ${t.message}", t)
                Toast.makeText(requireContext(), "Error: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }


    private fun loadMarkers(newsList: List<News>) {
        googleMap.clear()
        markerNewsMap.clear()

        for (news in newsList) {
            try {
                val coordinates = news.location?.coordinates

                if (coordinates == null || coordinates.size < 2) {
                    Log.d("MapFragment", "Skipping news with missing coordinates: ${news.title}")
                    continue
                }

                val latitude = coordinates[1]
                val longitude = coordinates[0]

                Log.d(
                    "MapFragment",
                    "Processing news: ${news.title}, Lat: $latitude, Long: $longitude"
                )

                if (latitude == 0.0 && longitude == 0.0) {
                    Log.d("MapFragment", "Skipping news with default location: ${news.title}")
                    continue
                }

                val location = LatLng(latitude, longitude)
                val marker = googleMap.addMarker(
                    MarkerOptions()
                        .position(location)
                        .title(news.title)
                )
                marker?.let {
                    markerNewsMap[it] = news
                    Log.d(
                        "MapFragment",
                        "Added marker: ${news.title} at Lat = $latitude, Long = $longitude"
                    )
                }
            } catch (e: Exception) {
                Log.e("MapFragment", "Error processing news: ${news.title}, Error: ${e.message}")
            }
        }

        if (markerNewsMap.isNotEmpty()) {
            val firstMarker = markerNewsMap.keys.first()
            googleMap.moveCamera(CameraUpdateFactory.newLatLngZoom(firstMarker.position, 10f))
        } else {
            Log.d("MapFragment", "No valid locations to display on the map.")
            Toast.makeText(
                requireContext(),
                "No valid locations to display on the map.",
                Toast.LENGTH_SHORT
            ).show()
        }
    }


    private fun openDetailView(news: News) {
        val detailFragment = DetailViewFragment.newInstance(news)
        parentFragmentManager.beginTransaction()
            .replace(
                R.id.fragment_container,
                detailFragment
            )
            .addToBackStack(null)
            .commit()
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
