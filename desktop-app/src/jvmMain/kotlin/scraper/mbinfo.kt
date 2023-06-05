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

fun _mbinfo(n: Int): List<INews> {
    val news: MutableList<INews> = mutableListOf()

    val options = ChromeOptions()
    options.addArguments("--headless")
    val browser: WebDriver = ChromeDriver(options)

    browser.get("https://mariborinfo.com/lokalno")
    println("Opened website: https://mariborinfo.com/lokalno")

    val links: List<WebElement> = browser.findElements(By.cssSelector("a[href^=\"/novica/\"]"))
    println(links.toString())

    for (i in 0 until links.size.coerceAtMost(n)) {
        val link: WebElement = links[i]
        val titleElement = link.findElement(By.cssSelector("span.title__value"))
        val titleText: String = titleElement.getAttribute("innerText")
        val title: String = titleText.trim()
        println("Title: $title")

        val url: String = link.getAttribute("href")
        println("URL: $url")

        browser.get(url)
        println("Opened article: $url")

        val wait = WebDriverWait(browser, Duration.ofSeconds(1))

        val authors: List<String> = browser.findElements(By.cssSelector(".username__name")).map {
            val authorName = it.getAttribute("innerText").split("/")[0].trim()
            println("Author: $authorName")
            authorName
        }

        val currentDate: Date = Date()
        val date: Date = Date()
        println("Date: $date")

        val content = browser.findElement(By.cssSelector("p"))
            .getAttribute("innerText").trim()
        println("Content: $content")

        val categories: List<String> =
            browser.findElements(By.cssSelector("a[href*=\"/tags/\"]"))
                .map {
                    val category = it.getAttribute("innerText").trim()
                    println("Category: $category")
                    category
                }

        news.add(
            INews(
                title = title,
                url = url,
                date = date,
                authors = authors,
                content = content,
                categories = categories,
                location = Location(
                    type = "Point",
                    coordinates = Pair(0.0, 0.0),
                )
            )
        )

        // Go back to the previous page to ensure elements are still valid
        browser.navigate().back()
    }

    browser.quit()
    return news
}
