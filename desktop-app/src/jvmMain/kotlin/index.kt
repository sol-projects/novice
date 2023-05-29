package org.example
import App
import _24ur
import _dnevnik
import _ekipa24
import _mbinfo
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import deleteNews
import getRtvSlo
import getZurnal24Slo
import getServisSta
import org.example.model.INews
import org.example.scraper.gov
import org.example.scraper.gov_vlada
import sendGet
import sendPost
import updateNews

fun main() = application {
    val news: ArrayList<INews> = sendGet()
    news.addAll(gov(2))

    Window(onCloseRequest = ::exitApplication) {
        App({ /* No-op */ }, news)
    }
    val jsonData = """
    {
        "title": "Sample Title",
        "url": "http://example.com/sample",
        "date": "2023-05-27T10:00:00.000Z",
        "authors": ["Author 1", "Author 2"],
        "content": "Sample content",
        "categories": ["Category 1", "Category 2"],
        "location": {
            "type": "Point",
            "coordinates": [12.345, 67.890]
        }
    }
""".trimIndent()

    val newjsonData = """
    {
        "title": "Bre",
        "url": "http://example.com/sample",
        "date": "2023-05-27T10:00:00.000Z",
        "authors": ["Author 1", "Author 2"],
        "content": "Sample content",
        "categories": ["Category 1", "Category 2"],
        "location": {
            "type": "Point",
            "coordinates": [12.345, 67.890]
        }
    }
""".trimIndent()

    sendPost(jsonData)
    //updateNews("6473095561a30cae2617f668",newjsonData)
    //deleteNews("6473095561a30cae2617f668")
    //_24ur(2)
    //println(gov(2))
    //println(gov_vlada(2))
    //_mbinfo(2)
    //_dnevnik(2)
    //_ekipa24(2)
}
