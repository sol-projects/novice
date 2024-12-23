package com.example.novinar

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import android.Manifest
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices

class PostNewsFragment : Fragment() {
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var userLocation: Location? = null
    private var editIndex: Int? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_post_news, container, false)
        val titleInput: EditText = view.findViewById(R.id.editTextTitle)
        val contentInput: EditText = view.findViewById(R.id.editTextContent)
        val postButton: Button = view.findViewById(R.id.buttonPost)

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(requireActivity())
        fetchUserLocation()

        arguments?.let {
            editIndex = it.getInt("news_index", -1)
            titleInput.setText(it.getString("news_title", ""))
            contentInput.setText(it.getString("news_content", ""))
        }

        postButton.setOnClickListener {
            val title = titleInput.text.toString()
            val content = contentInput.text.toString()
            val location = userLocation

            if (title.isNotEmpty() && content.isNotEmpty()) {
                if (editIndex != null && editIndex != -1) {
                    NewsRepository.editNews(
                        editIndex!!,
                        News(title, content, location?.latitude ?: 0.0, location?.longitude ?: 0.0)
                    )
                    Toast.makeText(context, "News updated successfully!", Toast.LENGTH_SHORT).show()
                } else {
                    NewsRepository.addNews(
                        News(title, content, location?.latitude ?: 0.0, location?.longitude ?: 0.0)
                    )
                    Toast.makeText(context, "News posted successfully!", Toast.LENGTH_SHORT).show()
                }

                requireActivity().supportFragmentManager.popBackStack()
                sendNotification(title, content)
                redirectToViewNewsFragment()
            } else {
                Toast.makeText(context, "Please fill out all fields", Toast.LENGTH_SHORT).show()
            }
        }

        return view
    }

    private fun fetchUserLocation() {
        if (ActivityCompat.checkSelfPermission(
                requireContext(), Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                1001
            )
            return
        }

        fusedLocationClient.lastLocation.addOnSuccessListener { location: Location? ->
            if (location != null) {
                userLocation = location
            } else {
                Toast.makeText(context, "Unable to retrieve location", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun sendNotification(title: String, content: String) {
        val intent = Intent(requireContext(), MainActivity::class.java).apply {
            putExtra("fragment", "ViewNewsFragment")
        }
        val pendingIntent = PendingIntent.getActivity(
            requireContext(),
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        NotificationHelper.sendNotification(
            requireContext(),
            "New Article Posted",
            "Title: $title\nContent: $content",
            pendingIntent
        )
    }

    private fun redirectToViewNewsFragment() {
        parentFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, ViewNewsFragment())
            .commit()
    }
}
