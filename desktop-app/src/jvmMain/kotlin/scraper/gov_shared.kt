import io.netty.util.concurrent.Promise
import org.example.model.INews
import org.openqa.selenium.By
import org.openqa.selenium.WebDriver
import org.openqa.selenium.WebElement
import org.openqa.selenium.chrome.ChromeDriver
import org.openqa.selenium.chrome.ChromeOptions
import java.util.Date

fun gov_shared(n: Int, website: String): List<INews> {
    val news: MutableList<INews> = mutableListOf()

    val options = ChromeOptions()
    options.addArguments("--headless=new")
    val driver: WebDriver = ChromeDriver(options)
    driver.get(website)
    val elements = driver.findElements(By.cssSelector(".title a")).map { element -> element.getAttribute("href") }

    for (i in 0 until n) {
        val url = elements[i]

        if (url.isEmpty()) {
            println("Cannot get url")
            continue
        }

        val newsItem = getNewspage(driver, "$url")
        news.add(newsItem)

        if (i == n - 1) {
            break
        }
    }

    driver.quit()
    return news
}

fun getNewspage(driver: WebDriver, url: String): INews {
    try {
        driver.get(url)

        val authorElement = driver.findElement(By.cssSelector(".organisations a"))
        val author: String = authorElement.text.trim()
        if (author.isEmpty()) {
            println("Cannot fetch author from page: $url")
        }

        val contentElements = driver.findElements(By.cssSelector(".content.col.left.grid-col-8 > :not(:first-child)"))
        val content = contentElements.joinToString("\n") { it.text.trim() }
        if (content.isEmpty()) {
            println("Cannot fetch content from page: $url")
        }

        val titleElement = driver.findElement(By.cssSelector(".page-head h1"))
        val title: String = titleElement.text.trim()

        val dateElement = driver.findElement(By.cssSelector(".info time"))
        val dateUnparsed: String = dateElement.text.trim()
        if (dateUnparsed.isEmpty()) {
            println("Cannot get date")
        }
        val dateSplit = dateUnparsed.split(". ")
        val date = Date(
            dateSplit[2].toInt(),
            dateSplit[1].toInt() - 1,
            dateSplit[0].toInt() + 1,
            0,
            0,
            0
        )

        return INews(
            title.trim(),
            url,
            date,
            listOf(author),
            content,
            emptyList(),
            ""
        )
    } catch (error: Exception) {
        println("Cannot fetch page: $url")
        return INews("", url, Date(), emptyList(), "", emptyList(), "")
    }
}
