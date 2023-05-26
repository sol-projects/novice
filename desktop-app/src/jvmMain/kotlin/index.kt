package org.example
import App
import _24ur
import _dnevnik
import _ekipa24
import _mbinfo
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import getRtvSlo
import getZurnal24Slo
import getServisSta
import org.example.model.INews
import org.example.scraper.gov
import org.example.scraper.gov_vlada

fun main() = application {
    val news: ArrayList<INews> = ArrayList()
    news.addAll(gov(2))

    Window(onCloseRequest = ::exitApplication) {
        App({ /* No-op */ }, news)
    }

    //_24ur(2)
    //println(gov(2))
    //println(gov_vlada(2))
    //_mbinfo(2)
    //_dnevnik(2)
    //_ekipa24(2)
}
