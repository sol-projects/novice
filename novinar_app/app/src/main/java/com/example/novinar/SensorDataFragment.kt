package com.example.novinar

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment

class SensorDataFragment : Fragment() {
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_sensor_data, container, false)
        val sensorDataTextView: TextView = view.findViewById(R.id.sensorDataTextView)

        val sensorData = SensorManager.getSensorData()
        val formattedData = sensorData.entries.joinToString("\n") { "${it.key}: ${it.value}" }

        sensorDataTextView.text = formattedData

        return view
    }
}
