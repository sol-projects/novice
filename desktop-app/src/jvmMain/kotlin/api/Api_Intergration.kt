import com.google.gson.Gson
import org.example.model.INews
import org.example.model.Location
import org.json.JSONArray
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.ArrayList
import com.mongodb.*
import com.mongodb.client.MongoClient
import com.mongodb.client.MongoClients
import com.mongodb.client.MongoDatabase
import com.google.gson.JsonObject
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException

fun sendGet(): ArrayList<INews> {
    val url = URL("http://localhost:8000/news")
    val rawData = url.readText()
    val jsonArray = JSONArray(rawData)

    val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
    dateFormat.timeZone = TimeZone.getTimeZone("UTC")
    val newsList = mutableListOf<INews>()

    for (i in 0 until jsonArray.length()) {
        val jsonObject = jsonArray.getJSONObject(i)

        val locationObj = jsonObject.getJSONObject("location")
        val location = Location(
            locationObj.getString("type"),
            Pair(locationObj.getJSONArray("coordinates").getDouble(0),
                locationObj.getJSONArray("coordinates").getDouble(1))
        )

        val authorsArray = jsonObject.getJSONArray("authors")
        val authorsList = mutableListOf<String>()
        for (j in 0 until authorsArray.length()) {
            authorsList.add(authorsArray.getString(j))
        }

        val categoriesArray = jsonObject.getJSONArray("categories")
        val categoriesList = mutableListOf<String>()
        for (j in 0 until categoriesArray.length()) {
            categoriesList.add(categoriesArray.getString(j))
        }

        val news = INews(
            title = jsonObject.getString("title"),
            url = jsonObject.getString("url"),
            date = dateFormat.parse(jsonObject.getString("date")),
            authors = authorsList,
            content = jsonObject.getString("content"),
            categories = categoriesList,
            location = location,
            _id = jsonObject.getString("_id"),
            __v = jsonObject.getInt("__v")
        )

        newsList.add(news)
    }

    return newsList as ArrayList<INews>
}



fun sendPost(data: String) {
    val url = URL("http://localhost:8000/news")
    val connection = url.openConnection() as HttpURLConnection

    connection.requestMethod = "POST"
    connection.setRequestProperty("Content-Type", "application/json")
    connection.doOutput = true

    val outputStream = OutputStreamWriter(connection.outputStream)
    outputStream.write(data)
    outputStream.flush()

    val responseCode = connection.responseCode
    println("Response Code: $responseCode")

    connection.disconnect()
}

fun updateNews(id: String, updatedNewsJson: String) {
    val client = OkHttpClient()

    val requestBody = updatedNewsJson.toRequestBody("application/json; charset=utf-8".toMediaType())

    val request = Request.Builder()
        .url("http://localhost:8000/news/$id")
        .put(requestBody)
        .build()

    client.newCall(request).execute().use { response ->
        if (!response.isSuccessful) throw IOException("Unexpected code $response")
        println(response.body?.string())
    }
}


fun deleteNews(id: String) {
    val client = OkHttpClient()

    val request = Request.Builder()
        .url("http://localhost:8000/news/$id")
        .delete()
        .build()

    client.newCall(request).execute().use { response ->
        if (!response.isSuccessful) throw IOException("Unexpected code $response")
        println(response.body?.string())
    }
}

