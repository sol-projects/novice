package org.example
import App
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import org.example.model.INews
import org.example.scraper.gov
import sendGet

fun main() = application {
    val news: ArrayList<INews> = sendGet()
    news.addAll(gov(2))

    Window(onCloseRequest = ::exitApplication) {
        App(news)
    }

    //sendPost(jsonData)
    //updateNews("6473095561a30cae2617f668",newjsonData)
    //deleteNews("6473095561a30cae2617f668")
    //_24ur(2)
    //println(gov(2))
    //println(gov_vlada(2))
    //_mbinfo(2)
    //_dnevnik(2)
    //_ekipa24(2)
}
