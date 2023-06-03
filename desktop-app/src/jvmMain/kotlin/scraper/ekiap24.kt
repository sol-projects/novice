import org.example.model.INews
import org.example.model.Location
import org.openqa.selenium.By
import org.openqa.selenium.WebDriver
import org.openqa.selenium.WebElement
import org.openqa.selenium.chrome.ChromeDriver
import org.openqa.selenium.chrome.ChromeOptions
import org.openqa.selenium.support.ui.WebDriverWait
import java.time.Duration
import java.util.Date

fun _ekipa24(n: Int): List<INews> {
    val news: MutableList<INews> = mutableListOf()

    val options = ChromeOptions()
    options.addArguments("--headless")
    val browser: WebDriver = ChromeDriver(options)

    browser.get("https://ekipa.svet24.si/")

    val links: List<WebElement> = browser.findElements(By.cssSelector("a[href^=\"/clanek\"]"))
    for (i in 0 until links.size.coerceAtMost(n)) {
        val link: WebElement = links[i]
        val url: String = link.getAttribute("href")

        browser.get("ekipa.svet24.si$url")

        val wait = WebDriverWait(browser, Duration.ofSeconds(5))

        val authors: List<String> = browser.findElements(By.cssSelector(".top-author"))
            .map { it.getAttribute("innerText").trim().split("\n")[0] }

        val content = browser.findElement(By.cssSelector("p"))
            .getAttribute("innerText").trim()
        val coords: Pair<Double, Double> = Pair(0.0, 0.0)
        val categoryLinks = browser.findElements(By.cssSelector("a[href^=\"/iskanje\"]"))
        val categories = categoryLinks.map { it.getAttribute("innerText").trim() }
        val titleText: String = "test"
        news.add(
            INews(
                title = titleText,
                url = "https://ekipa.svet24.si$url",
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
    }

    browser.quit()

    return news
}
