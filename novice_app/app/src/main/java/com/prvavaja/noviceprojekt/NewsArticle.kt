package com.prvavaja.noviceprojekt

import io.realm.kotlin.ext.realmListOf
import io.realm.kotlin.types.EmbeddedRealmObject
import io.realm.kotlin.types.RealmInstant
import io.realm.kotlin.types.RealmList
import io.realm.kotlin.types.RealmObject
import io.realm.kotlin.types.annotations.PrimaryKey
import java.text.SimpleDateFormat
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.mongodb.kbson.ObjectId
import java.util.Date

class NewsArticle(var title: String = "",
                  var url: String = "",
                  var date: java.util.Date = Date(),
                  var authors: List<String> = listOf(),
                  var content: String = "",
                  var categories: List<String> = listOf(),
                  var location: Location = Location(type = "Point", coordinates = Pair(0.0, 0.0)),
                  var _id: String = "0",
                  val __v: Int = 0) {
    override fun toString(): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        val formattedDate = dateFormat.format(date)
        val formattedAuthors = authors.joinToString("\", \"", "[\"", "\"]")
        val formattedCategories = categories.joinToString("\", \"", "[\"", "\"]")
        val coordinates = "[${location.coordinates.first}, ${location.coordinates.second}]"
        val jsonContent = Json.encodeToString(content)
        return "{\n" +
                "    \"title\": \"$title\",\n" +
                "    \"url\": \"$url\",\n" +
                "    \"date\": \"$formattedDate\",\n" +
                "    \"authors\": $formattedAuthors,\n" +
                "    \"content\": $jsonContent,\n" +
                "    \"categories\": $formattedCategories,\n" +
                "    \"location\": {\n" +
                "        \"type\": \"${location.type}\",\n" +
                "        \"coordinates\": $coordinates\n" +
                "    }\n" +
                "}"
    }
}
data class Location(
    var type: String,
    var coordinates: Pair<Double, Double>
)

open class NewsArticleRealm : RealmObject {
    @PrimaryKey
    var _id: ObjectId = ObjectId()
    var title: String = ""
    var url: String = ""
    var date: RealmInstant = RealmInstant.from(0,0)
    var views: RealmList<RealmInstant> = realmListOf()
    var authors: RealmList<String> = realmListOf()
    var content: String = ""
    var categories: RealmList<String> = realmListOf()
    var location: LocationRealm? = null
    var __v: Int = 0
}

open class LocationRealm : EmbeddedRealmObject {
    var coordinates: RealmList<Double> = realmListOf()
    var type: String = ""
}

fun fromRealmNewsArticle(newsArticleRealm: NewsArticleRealm): NewsArticle {
    val authors = newsArticleRealm.authors.toList()
    val categories = newsArticleRealm.categories.toList()
    val location = newsArticleRealm.location?.let {
        Location(
            type = it.type,
            coordinates = Pair(it.coordinates[0], it.coordinates[1])
        )
    }
    return NewsArticle(
        title = newsArticleRealm.title,
        url = newsArticleRealm.url,
        date = Date(newsArticleRealm.date.epochSeconds),
        authors = authors,
        content = newsArticleRealm.content,
        categories = categories,
        location = location ?: Location("", Pair(0.0, 0.0)),
        _id = newsArticleRealm._id.toString(),
        __v = newsArticleRealm.__v
    )
}
