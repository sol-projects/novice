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

import android.os.Handler
import android.os.Looper
import android.widget.EditText
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class SensorDataFragment : Fragment() {

    private val apiService: ApiService by lazy { RetrofitClient.apiService }
    private var fakeNewsHandler: Handler? = null
    private var fakeNewsTask: Runnable? = null
    private var isPostingFakeNews = false

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_sensor_data, container, false)
        val sensorDataTextView: TextView = view.findViewById(R.id.sensorDataTextView)
        val generateNewsButton: Button = view.findViewById(R.id.generateNewsButton)
        val deleteNewsButton: Button = view.findViewById(R.id.deleteNewsButton)
        val intervalInput: EditText = view.findViewById(R.id.intervalInput)
        val countInput: EditText = view.findViewById(R.id.countInput)
        val startFakeNewsButton: Button = view.findViewById(R.id.startFakeNewsButton)
        val stopFakeNewsButton: Button = view.findViewById(R.id.stopFakeNewsButton)

        val sensorData = SensorManager.getSensorData()
        val formattedData = sensorData.entries.joinToString("\n") { "${it.key}: ${it.value}" }
        sensorDataTextView.text = formattedData

        generateNewsButton.setOnClickListener { generateAndPostNews() }

        deleteNewsButton.setOnClickListener { deleteLast100News() }

        startFakeNewsButton.setOnClickListener {
            val interval = intervalInput.text.toString().toLongOrNull() ?: 60L
            val count = countInput.text.toString().toIntOrNull() ?: 100

            if (!isPostingFakeNews) {
                startPostingFakeNews(count, interval)
                Toast.makeText(requireContext(), "Started posting fake news!", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(requireContext(), "Fake news posting is already running!", Toast.LENGTH_SHORT).show()
            }
        }

        stopFakeNewsButton.setOnClickListener {
            stopPostingFakeNews()
            Toast.makeText(requireContext(), "Stopped posting fake news.", Toast.LENGTH_SHORT).show()
        }

        return view
    }

    private fun startPostingFakeNews(count: Int, interval: Long) {
        isPostingFakeNews = true
        fakeNewsHandler = Handler(Looper.getMainLooper())
        fakeNewsTask = object : Runnable {
            override fun run() {
                generateAndPostNews(count)
                fakeNewsHandler?.postDelayed(this, TimeUnit.SECONDS.toMillis(interval))
            }
        }
        fakeNewsHandler?.post(fakeNewsTask!!)
    }

    private fun stopPostingFakeNews() {
        isPostingFakeNews = false
        fakeNewsHandler?.removeCallbacks(fakeNewsTask!!)
        fakeNewsHandler = null
        fakeNewsTask = null
    }

    private fun generateAndPostNews(count: Int = 100) {
        val faker = Faker()

        for (i in 1..count) {
            val title = "Fake News ${faker.book().title()}"
            val content = "Generated Content: ${faker.lorem().paragraph()}\n" +
                    "Sensors: ${SensorManager.getSensorData()}"
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
        val timestampAsAuthor = System.currentTimeMillis().toString()

        val newsData = JSONObject().apply {
            put("title", title)
            put("content", content)
            put("categories", JSONArray().apply { put(category) })
            put("authors", JSONArray().apply { put(timestampAsAuthor) })
            put("url", "")
            put("location", JSONObject().apply {
                put("type", "Point")
                put("coordinates", JSONArray().apply {
                    put(longitude)
                    put(latitude)
                })
            })
        }

        Log.d("SensorDataFragment", "Fake News Payload: $newsData")

        val requestBody = RequestBody.create("application/json".toMediaTypeOrNull(), newsData.toString())

        apiService.postNews(requestBody).enqueue(object : Callback<Void> {
            override fun onResponse(call: Call<Void>, response: Response<Void>) {
                if (response.isSuccessful) {
                    Log.d("SensorDataFragment", "Fake News posted successfully: $title")
                } else {
                    Log.e(
                        "SensorDataFragment",
                        "Failed to post fake news: $title, Error: ${response.errorBody()?.string()}"
                    )
                }
            }

            override fun onFailure(call: Call<Void>, t: Throwable) {
                Log.e("SensorDataFragment", "Error posting fake news: ${t.localizedMessage}")
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
