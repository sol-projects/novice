import org.example.model.INews
import org.openqa.selenium.By
import org.openqa.selenium.WebDriver
import org.openqa.selenium.WebElement
import org.openqa.selenium.WindowType
import org.openqa.selenium.chrome.ChromeDriver
import org.openqa.selenium.chrome.ChromeOptions
import org.openqa.selenium.support.ui.ExpectedConditions
import org.openqa.selenium.support.ui.WebDriverWait
import java.time.Duration
import java.util.Date

fun _mbinfo(n: Int): List<INews> {
    val news: MutableList<INews> = mutableListOf()

    val options = ChromeOptions()
    options.addArguments("--headless=new")
    val browser: WebDriver = ChromeDriver(options)

    browser.get("https://mariborinfo.com/lokalno")

    val wait = WebDriverWait(browser, Duration.ofSeconds(5))
    //val element = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".leading-tight")));

    val links: List<WebElement> = browser.findElements(By.cssSelector("a[href^=\"/novica/\"]"))
    for (i in 0 until links.size.coerceAtMost(n)) {
        val link: WebElement = links[i]
        val titleElement = link.findElement(By.cssSelector("span.title__value"))
        val titleText: String = titleElement.getAttribute("innerText")
        //val altTitleElement = link.findElement(By.cssSelector("h1 span, h2 span, h3 span, h4 span, h5 span, h6 span"))
        //val altTitleText: String = altTitleElement.getAttribute("innerText")

        val title: String = titleText.trim()
        //if (title.isEmpty()) continue

        val url: String = "mariborinfo.com${link.getAttribute("href")}"

        browser.get(url)

        val wait = WebDriverWait(browser, Duration.ofSeconds(5))
        //val element = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".article__body")));

        val authors: List<String> = browser.findElements(By.cssSelector(".username__name")).map {
            it.getAttribute("innerText").split("/")[0].trim()
        }

        //val locationDateUnparsed: String =
          //  browser.findElement(By.cssSelector(".leading-caption")).getAttribute("innerText")
        //val locationDate: List<String> = locationDateUnparsed.split(", ")
        //val dateSplit: List<String> = locationDate[1].split(".")

        val currentDate: Date =  Date()
        val date :Date =  Date()
        //if (dateSplit[2].toInt() != currentDate.year) {
          //  browser.close()
            //continue
        //}
        //val date: Date = Date(dateSplit[2].toInt() - 1900, dateSplit[1].toInt() - 1, dateSplit[0].toInt() + 1)
        val contentElement = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".field.field--name-field-besedilo")))
        val content: String = contentElement.getAttribute("innerText")

       // val content: String = browser.findElement(By.cssSelector(".field.field--name-field-besedilo")).getAttribute("innerText")

        val categories: List<String> =
            browser.findElements(By.cssSelector("a[href*=\"/tags/\"]"))
                .map {
                    it.getAttribute("innerText").trim()
                }

        news.add(
            INews(
                title = title,
                url = url,
                date = date,
                authors = authors,
                content = content,
                categories = categories,
                location = "Maribor"
            )
        )

        browser.close()
    }

    browser.close()

    return news;
}