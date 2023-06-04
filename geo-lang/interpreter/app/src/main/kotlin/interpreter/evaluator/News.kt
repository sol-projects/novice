package interpreter.evaluator

import interpreter.tokenizer.TokenType
import com.google.gson.Gson
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.ArrayList
import com.google.gson.JsonArray
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import org.json.JSONArray
data class News(
    var title: String,
    var url: String,
    var date: Date,
    val authors: List<String>,
    var content: String,
    val categories: List<String>,
    val location: Location,
    val _id: String = "0",
    val __v: Int = 0
) {
    override fun toString(): String {
        val authors = authors.joinToString(", ")
        val categories = categories.joinToString(", ")
        return "News(title=$title, url=$url, date=$date, authors=$authors, categories=$categories location=$location)"
    }
}
data class Location(
    val type: String,
    val coordinates: Pair<Value.NumberType, Value.NumberType>
) {
    override fun toString() : String {
        return "Location(type='$type', coordinates=$coordinates)"
    }
}


fun fetch(): ArrayList<Value.NewsType> {
    val url = URL("http://localhost:8000/news")
    val rawData = url.readText()
    val jsonArray = JSONArray(rawData)

    val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
    dateFormat.timeZone = TimeZone.getTimeZone("UTC")
    val newsList = mutableListOf<Value.NewsType>()

    for (i in 0 until jsonArray.length()) {
        val jsonObject = jsonArray.getJSONObject(i)

        val locationObj = jsonObject.getJSONObject("location")
        val location = Location(
            locationObj.getString("type"),
            Pair(Value.NumberType(locationObj.getJSONArray("coordinates").getDouble(0)),
                Value.NumberType(locationObj.getJSONArray("coordinates").getDouble(1)))
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

        if(location.coordinates.first.value == 0.0 || location.coordinates.second.value == 0.0) {

        } else {
            val news = Value.NewsType(News(
                title = jsonObject.getString("title"),
                url = jsonObject.getString("url"),
                date = dateFormat.parse(jsonObject.getString("date")),
                authors = authorsList,
                content = jsonObject.getString("content"),
                categories = categoriesList,
                location = location,
                _id = jsonObject.getString("_id"),
                __v = jsonObject.getInt("__v")
            ))

            newsList.add(news)
        }
    }

    return newsList as ArrayList<Value.NewsType>
}