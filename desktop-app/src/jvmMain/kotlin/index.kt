package org.example
import App
import _24ur
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import getRtvSlo
import getZurnal24Slo
import getServisSta
import org.example.scraper.gov
import org.example.scraper.gov_vlada

fun main() = application {
    Window(onCloseRequest = ::exitApplication) {
        App()
    }

    _24ur(2)
    println(gov(2))
    println(gov_vlada(2))
}
