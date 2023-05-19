package org.example.scraper

import gov_shared
import org.example.model.INews

fun gov_vlada(n: Int): List<INews> {
    return gov_shared(n, "https://www.gov.si/drzavni-organi/vlada/novice/")
}