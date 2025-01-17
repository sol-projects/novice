package com.example.novinar

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import org.osmdroid.config.Configuration
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker

class MapFragment : Fragment() {

    private lateinit var mapView: MapView

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        Configuration.getInstance().load(requireContext(), requireActivity().getPreferences(0))
        val view = inflater.inflate(R.layout.fragment_map, container, false)

        mapView = view.findViewById(R.id.mapView)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(5.0)
        mapView.controller.setCenter(GeoPoint(50.0, 10.0))

        displayNewsMarkers()

        return view
    }

    private fun displayNewsMarkers() {
        // Fetch the list of news
        val newsList = NewsRepository.getNewsList() // Replace with your API or database call

        for (news in newsList) {
            // Skip markers with invalid coordinates
            if (news.latitude == 0.0 && news.longitude == 0.0) continue

            // Create a marker for the news
            val marker = Marker(mapView)
            marker.position = GeoPoint(news.latitude, news.longitude)
            marker.title = news.title
            //marker.snippet = news.content
            marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)

            // Open DetailViewFragment on long click
            marker.setOnMarkerClickListener { m, _ ->
                openDetailView(news)
                true
            }

            mapView.overlays.add(marker)
        }

        mapView.invalidate()
    }

    private fun openDetailView(news: News) {
        val detailFragment = DetailViewFragment.newInstance(news)
        parentFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, detailFragment)
            .addToBackStack(null)
            .commit()
    }

    override fun onResume() {
        super.onResume()
        mapView.onResume()
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
    }
}
