import org.example.model.INews
import org.example.model.Location
import org.openqa.selenium.By
import org.openqa.selenium.WebDriver
import org.openqa.selenium.WebElement
import org.openqa.selenium.chrome.ChromeDriver
import org.openqa.selenium.chrome.ChromeOptions
import java.text.SimpleDateFormat
import java.util.Date

fun getRtvSlo(numArticlesToOpen: Int): List<INews> {

    val options = ChromeOptions()
    options.addArguments("--headless=new");
    val driver: WebDriver = ChromeDriver(options)

    driver.get("https://www.rtvslo.si/novice")

    val articles: List<WebElement> = driver.findElements(By.className("md-news"))

    // Create an empty list to store the news articles
    val newsList = mutableListOf<INews>()

    // Loop through the specified number of news articles and extract the data
    for (i in 0 until numArticlesToOpen) {
        val article: WebElement = articles[i]

        val url: String = article.findElement(By.cssSelector("a")).getAttribute("href")
        driver.get(url)

        var metaTag: WebElement = driver.findElement(By.cssSelector("meta[name='title']"))
        val title: String = metaTag.getAttribute("content")

        metaTag = driver.findElement(By.cssSelector("meta[name='published_date']"))
        val dateString: String = metaTag.getAttribute("content")
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'")
        val date: Date = dateFormat.parse(dateString)

        val content: String = driver.findElement(By.cssSelector("article.article")).text

        val metaElements = driver.findElements(By.cssSelector("meta.elastic[name='author']"))
        val authors = metaElements.map { it.getAttribute("content") }

        val element = driver.findElement(By.cssSelector("div.place-source"))
        val location = element.text.trim().split(" - ")[0]

        metaTag = driver.findElement(By.cssSelector("meta.elastic[name='keywords']"))
        val categoriesString: String = metaTag.getAttribute("content")
        val categories = categoriesString.split(",")

        // Create an INews object and add it to the newsList
        val news = INews(title, url, date, authors, content, categories, Location(
            type = "Point",
            coordinates = Pair(0.0,0.0),
        )
        )
        newsList.add(news)

        driver.navigate().back()
    }

    driver.quit()

    return newsList
}
