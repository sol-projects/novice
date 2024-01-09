import org.example.model.INews
import org.example.model.Location
import org.openqa.selenium.By
import org.openqa.selenium.WebDriver
import org.openqa.selenium.WebElement
import org.openqa.selenium.chrome.ChromeDriver
import org.openqa.selenium.chrome.ChromeOptions
import java.text.SimpleDateFormat
import java.util.Date

fun extractCountryFromTitle(title: String): String {
    val countryPrefixes = listOf("v ", "v ", "na ")

    for (prefix in countryPrefixes) {
        val startIndex = title.indexOf(prefix)
        if (startIndex != -1) {
            val endIndex = title.indexOf(" ", startIndex + prefix.length)
            if (endIndex != -1) {
                return title.substring(startIndex + prefix.length, endIndex)
            }
        }
    }

    return ""
}

fun getServisSta(numArticlesToOpen: Int): List<INews> {

    val options = ChromeOptions()
    options.addArguments("--headless=new");
    val driver: WebDriver = ChromeDriver(options)

    driver.get("https://servis.sta.si/")


    val newsList = mutableListOf<INews>()

    for (i in 0 until numArticlesToOpen) {
        val articles: List<WebElement> = driver.findElements(By.className("item"))

        val article: WebElement = articles[i]

        val url: String = article.findElement(By.cssSelector("a")).getAttribute("href")
        driver.get(url)

        val title = driver.findElement(By.cssSelector("article.articleui h1")).text

        val articleElement = driver.findElement(By.tagName("article"))
        val leadElement = try {
            articleElement.findElement(By.className("lead"))
        } catch (e: NoSuchElementException) {
            null
        }
        val lead = leadElement?.text ?: ""

        val textElements = articleElement.findElements(By.className("text"))
        val preText = if (textElements.isNotEmpty()) {
            val preTextElement = textElements[0]
            val preElements = preTextElement.findElements(By.tagName("pre"))
            if (preElements.isNotEmpty()) {
                preElements[0].text
            } else {
                ""
            }
        } else {
            ""
        }

        val content = if (lead.isNotEmpty() && preText.isNotEmpty()) {
            "$lead $preText"
        } else if (lead.isNotEmpty()) {
            lead
        } else if (preText.isNotEmpty()) {
            preText
        } else {
            ""
        }

        val categoryElement = driver.findElement(By.cssSelector("aside.articlemeta div.items > div:nth-child(2)"))
        val categories = listOf(categoryElement.text.replace("Kategorija:", "").trim())

        val authorElement = driver.findElement(By.cssSelector("aside.articlemeta div.items > div:nth-child(4)"))
        val authors = authorElement.text.replace("Avtor:", "").trim().split("/")


        val location = extractCountryFromTitle(title)

        val date: Date = SimpleDateFormat("yyyy-MM-dd").parse("2000-10-10")

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
