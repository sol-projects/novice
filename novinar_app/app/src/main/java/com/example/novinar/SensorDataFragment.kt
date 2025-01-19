package com.example.novinar

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.example.novinar.api.ApiService
import com.example.novinar.api.RetrofitClient
import com.github.javafaker.Faker
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class SensorDataFragment : Fragment() {

    private val apiService: ApiService by lazy {
        RetrofitClient.apiService
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_sensor_data, container, false)
        val sensorDataTextView: TextView = view.findViewById(R.id.sensorDataTextView)
        val generateNewsButton: Button = view.findViewById(R.id.generateNewsButton)
        val deleteNewsButton: Button = view.findViewById(R.id.deleteNewsButton)

        val sensorData = SensorManager.getSensorData()
        val formattedData = sensorData.entries.joinToString("\n") { "${it.key}: ${it.value}" }
        sensorDataTextView.text = formattedData

        generateNewsButton.setOnClickListener {
            generateAndPostNews()
        }

        deleteNewsButton.setOnClickListener {
            deleteLast100News()
        }

        return view
    }

    private fun generateAndPostNews() {
        val faker = Faker()

        for (i in 1..100) {
            val title = faker.book().title()
            val content = faker.lorem().paragraph()
            val category = faker.book().genre()
            val latitude = faker.address().latitude().toDoubleOrNull() ?: 0.0
            val longitude = faker.address().longitude().toDoubleOrNull() ?: 0.0

            postNews(title, content, category, latitude, longitude)
        }
    }

    private fun postNews(
        title: String,
        content: String,
        category: String,
        latitude: Double,
        longitude: Double
    ) {
        val titlePart = RequestBody.create("text/plain".toMediaTypeOrNull(), title)
        val contentPart = RequestBody.create("text/plain".toMediaTypeOrNull(), content)
        val categoryPart = RequestBody.create("text/plain".toMediaTypeOrNull(), category)

        val locationJson = "{\"type\": \"Point\", \"coordinates\": [$longitude, $latitude]}"
        val locationPart = RequestBody.create("application/json".toMediaTypeOrNull(), locationJson)

        val imagePart: MultipartBody.Part? = null

        apiService.postNews(titlePart, contentPart, categoryPart, locationPart, imagePart)
            .enqueue(object : Callback<Void> {
                override fun onResponse(call: Call<Void>, response: Response<Void>) {
                    if (response.isSuccessful) {
                        Log.d("SensorDataFragment", "News posted successfully: $title")
                    } else {
                        Log.e(
                            "SensorDataFragment",
                            "Failed to post news: $title, Error: ${response.errorBody()?.string()}"
                        )
                    }
                }

                override fun onFailure(call: Call<Void>, t: Throwable) {
                    Log.e("SensorDataFragment", "Error posting news: ${t.localizedMessage}")
                }
            })
    }


    private fun deleteLast100News() {
        apiService.getNews().enqueue(object : Callback<List<News>> {
            override fun onResponse(call: Call<List<News>>, response: Response<List<News>>) {
                if (response.isSuccessful) {
                    val newsList = response.body()?.takeLast(100) ?: emptyList()

                    if (newsList.isEmpty()) {
                        Toast.makeText(requireContext(), "No news to delete!", Toast.LENGTH_SHORT)
                            .show()
                        return
                    }

                    deleteNewsBatch(newsList)
                } else {
                    Toast.makeText(requireContext(), "Failed to fetch news!", Toast.LENGTH_SHORT)
                        .show()
                }
            }

            override fun onFailure(call: Call<List<News>>, t: Throwable) {
                Toast.makeText(
                    requireContext(),
                    "Error fetching news: ${t.localizedMessage}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        })
    }

    private fun deleteNewsBatch(newsList: List<News>) {
        for (news in newsList) {
            news._id?.let { id ->
                apiService.deleteNews(id).enqueue(object : Callback<Void> {
                    override fun onResponse(call: Call<Void>, response: Response<Void>) {
                        if (response.isSuccessful) {
                            Log.d("SensorDataFragment", "Deleted news: ${news.title}")
                        } else {
                            Log.e("SensorDataFragment", "Failed to delete news: ${news.title}")
                        }
                    }

                    override fun onFailure(call: Call<Void>, t: Throwable) {
                        Log.e(
                            "SensorDataFragment",
                            "Error deleting news: ${news.title}, ${t.localizedMessage}"
                        )
                    }
                })
            }
        }

        Toast.makeText(requireContext(), "Last 100 news deleted!", Toast.LENGTH_SHORT).show()
    }
}
