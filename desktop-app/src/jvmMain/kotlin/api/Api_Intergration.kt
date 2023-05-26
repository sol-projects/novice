import org.example.model.INews
import org.example.model.Location
import org.json.JSONArray
import java.net.URL
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.ArrayList

fun sendGet(): ArrayList<INews> {
    val url = URL("http://108.143.49.11:8000/news")
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
