import org.example.model.INews
import org.example.model.Location
import org.openqa.selenium.By
import org.openqa.selenium.WebDriver
import org.openqa.selenium.WebElement
import org.openqa.selenium.chrome.ChromeDriver
import org.openqa.selenium.chrome.ChromeOptions
import org.openqa.selenium.support.ui.ExpectedConditions
import org.openqa.selenium.support.ui.WebDriverWait
import java.time.Duration
import java.util.Date

fun _dnevnik(n: Int): List<INews> {
    val news: MutableList<INews> = mutableListOf()

    val options = ChromeOptions()
    options.addArguments("--headless")
    val browser: WebDriver = ChromeDriver(options)

    browser.get("https://www.dnevnik.si/slovenija")

    val links: List<WebElement> = browser.findElements(By.cssSelector("a[href^=\"/104\"].view-more"))
    for (i in 0 until links.size.coerceAtMost(n)) {
        val link: WebElement = links[i]
        val url: String = link.getAttribute("href")

        if (url.startsWith("/104")) {
            browser.get("https://www.dnevnik.si$url")

            val wait = WebDriverWait(browser, Duration.ofSeconds(5))

            val authors: List<String> = browser.findElements(By.cssSelector(".article-source"))
                .map { it.getAttribute("innerText").split(",")[0].trim() }

            val dateElement = browser.findElement(By.cssSelector(".dtstamp"))
            val dateText: String = dateElement.getAttribute("innerText")
            val date: Date = Date(dateText) // Please handle date parsing according to the format

            val content = browser.findElement(By.cssSelector("article"))
                .getAttribute("innerText").trim()

            val firstSentence = content.split('\n')[0]
            val title = if (firstSentence.endsWith('.')) firstSentence else "$firstSentence."

            val categoryLinks = browser.findElements(By.cssSelector("a[href*=\"/tag/\"]"))
            val categories = categoryLinks.map { it.getAttribute("innerText").trim() }
            val coords: Pair<Double, Double> = Pair(0.0, 0.0)
            news.add(
                INews(
                    title = title.trim(),
                    url = "https://www.dnevnik.si$url",
                    date = Date(),
                    authors = authors,
                    content = content,
                    categories = categories,
                    location = Location(
                        type = "Point",
                        coordinates = coords,
                    )
                )
            )

            browser.switchTo().window(browser.windowHandles.first())
            browser.close()
        }
    }

    browser.quit()

    return news
}
