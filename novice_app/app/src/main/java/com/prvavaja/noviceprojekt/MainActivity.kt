package com.prvavaja.noviceprojekt

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Toast
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.prvavaja.noviceprojekt.databinding.ActivityMainBinding
import io.realm.kotlin.Realm
import io.realm.kotlin.ext.query
import io.realm.kotlin.mongodb.Credentials
import io.realm.kotlin.mongodb.subscriptions
import io.realm.kotlin.mongodb.sync.SyncConfiguration
import kotlinx.coroutines.runBlocking
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat

class MainActivity : AppCompatActivity(), MyAdapter.OnItemClickListener {
    private lateinit var binding: ActivityMainBinding //ADD THIS LINE
    private lateinit var myApplication:MyAplication
    private lateinit var adapter: MyAdapter
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater) //ADD THIS LINE
        setContentView(binding.root)
        myApplication = application as MyAplication

        //setContentView(R.layout.activity_main)





        binding.btnSetings.setOnClickListener {  val intent = Intent(this, SetingsActivity::class.java)
            startActivity(intent) }
        binding.buttonCamera.setOnClickListener {  val intent = Intent(this, CameraActivty::class.java)
            startActivity(intent) }
        binding.btnMessage.setOnClickListener {  val intent = Intent(this, MasageActivity::class.java)
            startActivity(intent) }

        /*runBlocking {
            try {
                myApplication.realm_app = io.realm.kotlin.mongodb.App.create("application-0-qcgjd")
                myApplication.user = myApplication.realm_app.login(Credentials.anonymous())

                val config = SyncConfiguration.Builder(
                    myApplication.user,
                    setOf(NewsArticleRealm::class, LocationRealm::class)
                ).build()

                myApplication.realm = Realm.open(config)

                myApplication.realm.subscriptions.update {
                    add(myApplication.realm.query<NewsArticleRealm>())
                }

                myApplication.realm.subscriptions.waitForSynchronization() // ce dobis Null Exception, je tu problem
                myApplication.realm.query<NewsArticleRealm>().find().map { fromRealmNewsArticle(it)}.forEach { list1.addItem(it) }

            } catch(e: Exception) { // Realm je malo nestabilen pa vcasih vrze cudne exceptione, zato belezim za vse
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                val exceptionAsString = sw.toString()

                println("Realm exception: $e\n$exceptionAsString")
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Invalid email or password", Toast.LENGTH_SHORT).show()
                }
                return@runBlocking
            }
        }*/
    }
    override fun onItemClick(novica: NewsArticle) {
        val intent = Intent(this, DisplayActivity::class.java)

        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        val formattedDate = dateFormat.format(novica.date)
        val formattedAuthors = novica.authors.joinToString("\", \"", "[\"", "\"]")
        val formattedCategories = novica.categories.joinToString("\", \"", "[\"", "\"]")
        val coordinates = "[${novica.location.coordinates.first}, ${novica.location.coordinates.second}]"
        // Add extra data to the intent to pass the UUID of the clicked Sadje object
        intent.putExtra("id", novica._id)
        intent.putExtra("title",novica.title)
        intent.putExtra("url", novica.url)
        intent.putExtra("date", formattedDate)
        intent.putExtra("authors",formattedAuthors)
        intent.putExtra("categories",formattedCategories)
        intent.putExtra("cordinates",coordinates)
        intent.putExtra("content",novica.content)
        intent.putExtra("__v",novica.__v)
        // Start the MainActivity using the created intent
        //startActivity(intent)
        startActivity(intent)
    }

    /*override fun onDestroy() {
        super.onDestroy()
        myApplication.realm.close()
    }*/
}