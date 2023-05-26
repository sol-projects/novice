import com.google.gson.Gson
import org.example.model.INews
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.URL
import okhttp3.OkHttpClient
import okhttp3.Request

fun sendGet() {
    val url = "http://localhost:8000/news" // Replace with your actual URL
    val client = OkHttpClient()


// Create a request object
    val request = Request.Builder()
        .url(url)
        .build()

// Execute the request
    val response = client.newCall(request).execute()

// Get the JSON response body as a string
    val jsonString = response.body?.string()

// Use Gson to parse the JSON string into a temporary object
    val gson = Gson()
    val temporaryINews = gson.fromJson(jsonString, Array<INews>::class.java)[0]


// Print the final INews object
    //println(INews)
    }

fun readJsonFromUrl(url: String): String {
    val connection = URL(url).openConnection()
    connection.connect()
    val reader = BufferedReader(InputStreamReader(connection.getInputStream()))
    val jsonString = reader.use { it.readText() }
    reader.close()
    return jsonString
}
