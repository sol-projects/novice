package com.example.novinar

import android.os.Bundle
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetDialog
import org.osmdroid.bonuspack.clustering.RadiusMarkerClusterer
import org.osmdroid.config.Configuration
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker

class MapFragment : Fragment() {

    private lateinit var mapView: MapView
    private lateinit var clusterer: RadiusMarkerClusterer

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

        clusterer = RadiusMarkerClusterer(requireContext())
        mapView.overlays.add(clusterer)

        displayNewsMarkers()

        mapView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                val tappedGeoPoint = mapView.projection.fromPixels(
                    event.x.toInt(),
                    event.y.toInt()
                )
                detectClusterTap(tappedGeoPoint as GeoPoint)
            }
            false
        }

        return view
    }

    private fun displayNewsMarkers() {
        val newsList = NewsRepository.getNewsList()

        for (news in newsList) {
            val marker = Marker(mapView)
            marker.position = GeoPoint(news.latitude, news.longitude)
            marker.title = news.title
            marker.snippet = "${news.content}\nPosted at: ${news.timestamp}" // Include timestamp
            marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)

            marker.setOnMarkerClickListener { m, _ ->
                showBottomSheet(listOf(news))
                m.showInfoWindow()
                true
            }

            clusterer.add(marker)
        }

        mapView.invalidate()
    }

    private fun detectClusterTap(tappedGeoPoint: GeoPoint) {
        val clusterItems = mutableListOf<News>()

        for (item in clusterer.items) {
            if (item is Marker) {
                val distance = tappedGeoPoint.distanceToAsDouble(item.position)
                if (distance < 50) {
                    clusterItems.add(
                        News(
                            _id = null, // ID is not used in this context
                            title = item.title,
                            content = item.snippet?.split("\n")?.get(0) ?: "",
                            latitude = item.position.latitude,
                            longitude = item.position.longitude,
                            timestamp = item.snippet?.split("\n")?.get(1)?.replace("Posted at: ", "") ?: ""
                        )
                    )
                }
            }
        }

        if (clusterItems.size > 1) {
            showBottomSheet(clusterItems)
        }
    }

    private fun showBottomSheet(newsList: List<News>) {
        val bottomSheetView = layoutInflater.inflate(R.layout.layout_bottom_sheet, null)

        val recyclerView: RecyclerView = bottomSheetView.findViewById(R.id.clusterRecyclerView)

        recyclerView.layoutManager = LinearLayoutManager(requireContext())
        recyclerView.adapter = NewsAdapter(newsList, onEdit = {}, onDelete = {}) // Add edit/delete logic if needed

        val dialog = BottomSheetDialog(requireContext())
        dialog.setContentView(bottomSheetView)

        dialog.show()
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
