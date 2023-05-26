import org.example.model.INews
import org.example.model.Location
import org.openqa.selenium.By
import org.openqa.selenium.WebDriver
import org.openqa.selenium.WebElement
import org.openqa.selenium.chrome.ChromeDriver
import org.openqa.selenium.chrome.ChromeOptions
import java.text.SimpleDateFormat
import java.util.*

fun getZurnal24Slo(numArticlesToOpen: Int): List<INews> {
    val options = ChromeOptions()
    options.addArguments("--headless=new");
    val driver: WebDriver = ChromeDriver(options)

    driver.get("https://www.zurnal24.si/")

    // Create an empty list to store the news articles
    val newsList = mutableListOf<INews>()

    // Loop through the specified number of news articles and extract the data
    for (i in 0 until numArticlesToOpen) {
        val articles: List<WebElement> = driver.findElements(By.className("card__wrap"))

        val article: WebElement = articles[i]
        val url: String = article.findElement(By.className("card__link")).getAttribute("href")
        driver.get(url)

        val title: String = driver.findElement(By.className("article__title")).text

        val metaTag = driver.findElement(By.cssSelector("meta[itemprop='datePublished']"))
        val dateString: String = metaTag.getAttribute("content")
        val dateFormat = SimpleDateFormat("yyyy-MM-dd")
        val date: Date = dateFormat.parse(dateString)

        val paddingWrapper = driver.findElement(By.cssSelector("div.article__content.cf"))
        val elements = paddingWrapper.findElements(By.cssSelector("p, h2"))
        var content = ""
        for (element in elements) {
            content += element.text
        }

        val sectionLinks = driver.findElements(By.className("article__sections_link"))   // fix this to get it from article__tag_name
        val categories = sectionLinks.map { it.text }

        val authorsDiv = driver.findElement(By.cssSelector("div.article__authors"))
        val authorLinks = authorsDiv.findElements(By.cssSelector("a"))
        val authors = authorLinks.map { it.text }

        /*
        val element = driver.findElement(By.cssSelector("div.place-source"))
        val location = element.text.trim().split(" - ")[0]
        */

        // Create an INews object and add it to the newsList
        val news = INews(title, url, date, authors, content, categories, Location(
            type = "Point",
            coordinates = Pair(0.0,0.0),
        )
        )
        newsList.add(news)

        println("Title: $title")
        println("Author: $authors")
        println("Date: $date")
        //println("Location: $location")
        println("Content: $content")
        println("Categories: $categories")
        println("URL: $url")
        println()

        driver.navigate().back() // Navigate back to the main page
    }

    driver.quit()

    return newsList
}
