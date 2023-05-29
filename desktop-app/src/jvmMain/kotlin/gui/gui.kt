import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.ClickableText
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.List
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.*
import org.example.model.INews
import org.example.model.Location
import org.example.scraper.gov
import org.example.scraper.gov_vlada

enum class Section {
    DataBaseCollection,
    AddNews,
    Scrape
}

enum class Scraper {
    _24ur,
    _gov,
    _gov_vlada,
    //_mbinfo,
    _dnevnik,
    _ekipa24,
    _servisSta,
    _zurnal24,
    _rtvSlo,
    scrapeAll
}
@Composable
fun Header(onSelectionChanged: (Section) -> Unit) {
    TopAppBar(
        modifier = Modifier
            .fillMaxWidth()
            .height(26.dp)
            .background(MaterialTheme.colors.primary)
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 16.dp)
                .fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row(
                modifier = Modifier
                    .clickable {
                        onSelectionChanged(Section.DataBaseCollection)
                    }
                    .weight(0.33f),
                horizontalArrangement = Arrangement.Center
            ) {
                Icon(
                    imageVector = Icons.Default.List,
                    contentDescription = "Data base collection",
                    tint = Color.White,
                    modifier = Modifier
                        .size(24.dp)
                        .padding(end = 8.dp),
                )
                Text(
                    text = "Data base collection",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                )
            }
            Row(
                modifier = Modifier
                    .clickable {
                        onSelectionChanged(Section.AddNews)
                    }
                    .weight(0.33f),
                horizontalArrangement = Arrangement.Center

            ) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = "Add news",
                    tint = Color.White,
                    modifier = Modifier
                        .size(24.dp)
                        .padding(end = 8.dp)
                )
                Text(
                    text = "Add news",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                )
            }
            Row(
                modifier = Modifier
                    .clickable {
                        onSelectionChanged(Section.Scrape)
                    }
                    .weight(0.34f),
                horizontalArrangement = Arrangement.Center

            ) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = "Scrape",
                    tint = Color.White,
                    modifier = Modifier
                        .size(24.dp)
                        .padding(end = 8.dp)
                )
                Text(
                    text = "Scrape",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}

@Composable
fun Footer(selected: Section, onFilterClicked: () -> Unit, onOrderByClicked: () -> Unit) {
    BottomAppBar(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colors.primary)
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 16.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (selected == Section.DataBaseCollection) {
                Row(
                    modifier = Modifier.weight(0.2f),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Button(
                        onClick = onFilterClicked
                    ) {
                        Text("Filter")
                    }
                    Button(
                        onClick = onOrderByClicked
                    ) {
                        Text("Order By")
                    }
                }
            }
        }
    }
}


@Composable
fun NewsRow(news: INews, onDeleteClicked: () -> Unit, onEditClicked: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 8.dp),
        elevation = 4.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = news.title,
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.subtitle1,
                    modifier = Modifier.weight(1f)
                )
                Row {
                    IconButton(
                        onClick = onDeleteClicked
                    ) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Delete",
                            tint = Color.Red
                        )
                    }
                    IconButton(
                        onClick = onEditClicked
                    ) {
                        Icon(
                            imageVector = Icons.Default.Edit,
                            contentDescription = "Edit",
                            tint = Color.Blue
                        )
                    }
                }
            }
            Text(
                text = "URL: ${news.url}",
                style = MaterialTheme.typography.body1,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Text(
                text = "Date: ${SimpleDateFormat("yyyy-MM-dd").format(news.date)}",
                style = MaterialTheme.typography.body1,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Text(
                text = "Authors: ${news.authors.joinToString(", ")}",
                style = MaterialTheme.typography.body1,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Text(
                text = "Content: ${limitContentTo500Characters(news.content)}",
                style = MaterialTheme.typography.body1,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Text(
                text = "Categories: ${news.categories.joinToString(", ")}",
                style = MaterialTheme.typography.body1,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Text(
                text = "Location: ${news.location}",
                style = MaterialTheme.typography.body1
            )
        }
    }
}


@Composable
fun limitContentTo500Characters(content: String): String {
    return if (content.length <= 500) {
        content
    } else {
        content.substring(0, 500) + "..."
    }
}
@Composable
fun ScrapeSection() {
    var expandedScraper by remember { mutableStateOf(false) }
    var expandedNumber by remember { mutableStateOf(false) }
    var selectedScraper by remember { mutableStateOf<Scraper?>(null) }
    var selectedNumber by remember { mutableStateOf(0) }
    val scraperOptions = Scraper.values().map { it.name.replace("_", " ") }
    val newsScraped = remember { mutableStateListOf<INews>() }


    Column(modifier = Modifier.padding(16.dp)) {
        Text(
            text = "Scrape Section",
            style = MaterialTheme.typography.h5,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        // Select Scraper dropdown
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 8.dp)) {
            Text(
                text = "Select Scraper:",
                style = MaterialTheme.typography.subtitle1,
                fontWeight = FontWeight.Bold
            )

            Box(modifier = Modifier.padding(start = 8.dp)) {
                ClickableText(
                    text = buildAnnotatedString {
                        if (selectedScraper != null) {
                            append(selectedScraper!!.name.replace("_", " "))
                        } else {
                            withStyle(style = SpanStyle(fontWeight = FontWeight.Bold)) {
                                append("Select a scraper")
                            }
                        }
                    },
                    onClick = {
                        expandedScraper = true
                    }
                )

                DropdownMenu(
                    expanded = expandedScraper,
                    onDismissRequest = {
                        expandedScraper = false
                    },
                    modifier = Modifier.width(200.dp)
                ) {
                    scraperOptions.forEachIndexed { index, scraper ->
                        DropdownMenuItem(
                            onClick = {
                                selectedScraper = Scraper.values()[index]
                                expandedScraper = false
                            }
                        ) {
                            Text(text = scraper)
                        }
                    }
                }
            }
        }

        // Select Number dropdown
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 8.dp)) {
            Text(
                text = "Select Number:",
                style = MaterialTheme.typography.subtitle1,
                fontWeight = FontWeight.Bold
            )

            Box(modifier = Modifier.padding(start = 8.dp)) {
                ClickableText(
                    text = buildAnnotatedString {
                        if (selectedNumber != 0) {
                            append(selectedNumber.toString())
                        } else {
                            withStyle(style = SpanStyle(fontWeight = FontWeight.Bold)) {
                                append("Enter a number")
                            }
                        }
                    },
                    onClick = {
                        expandedNumber = true
                    }
                )

                DropdownMenu(
                    expanded = expandedNumber,
                    onDismissRequest = {
                        expandedNumber = false
                    },
                    modifier = Modifier.width(200.dp)
                ) {
                    DropdownMenuItem(
                        onClick = {
                            selectedNumber = 1
                            expandedNumber = false
                        }
                    ) {
                        Text(text = "Enter a number")
                    }

                    (1..10).forEach { number ->
                        DropdownMenuItem(
                            onClick = {
                                selectedNumber = number
                                expandedNumber = false
                            }
                        ) {
                            Text(text = number.toString())
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                if (selectedScraper != null && selectedNumber != 0) {
                    when (selectedScraper) {
                        Scraper._24ur -> newsScraped.addAll(_24ur(selectedNumber))
                        Scraper._gov -> newsScraped.addAll(gov(selectedNumber))
                        Scraper._gov_vlada -> newsScraped.addAll(gov_vlada(selectedNumber))
                        Scraper._dnevnik -> newsScraped.addAll(_dnevnik(selectedNumber))
                        Scraper._ekipa24 -> newsScraped.addAll(_ekipa24(selectedNumber))
                        //Scraper._mbinfo -> newsScraped.addAll(_mbinfo(selectedNumber))
                        Scraper._servisSta -> newsScraped.addAll(getServisSta(selectedNumber))
                        Scraper._zurnal24 -> newsScraped.addAll(getZurnal24Slo(selectedNumber))
                        Scraper._rtvSlo -> newsScraped.addAll(getRtvSlo(selectedNumber))
                        Scraper.scrapeAll -> {
                            newsScraped.addAll(_24ur(selectedNumber))
                            newsScraped.addAll(gov(selectedNumber))
                            newsScraped.addAll(gov_vlada(selectedNumber))
                            newsScraped.addAll(_dnevnik(selectedNumber))
                            newsScraped.addAll(_ekipa24(selectedNumber))
                            //newsScraped.addAll(_mbinfo(selectedNumber))
                            newsScraped.addAll(getServisSta(selectedNumber))
                            newsScraped.addAll(getZurnal24Slo(selectedNumber))
                            newsScraped.addAll(getRtvSlo(selectedNumber))
                        }

                        else -> {}
                    }
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = "Scrape")
        }

        Spacer(modifier = Modifier.height(12.dp))

        if (newsScraped.isNotEmpty()) {
            Text(
                text = "Scraped News:",
                style = MaterialTheme.typography.subtitle1,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            LazyColumn(modifier = Modifier.weight(1f)) {
                items(newsScraped) { news ->
                    scrapedShow(news) {
                        newsScraped.remove(news)
                    }
                }
            }
        }
    }
}
@Composable
fun scrapedShow(news: INews, onRemoveClicked: () -> Unit) {
    var editedNews by remember { mutableStateOf(news) }
    var isEditing by remember { mutableStateOf(false) }

    if (isEditing) {
        Column(modifier = Modifier.padding(16.dp)) {
            TextField(
                value = editedNews.title,
                onValueChange = { editedNews = editedNews.copy(title = it) },
                label = { Text("Title") },
                modifier = Modifier.fillMaxWidth()
            )
            TextField(
                value = editedNews.url,
                onValueChange = { editedNews = editedNews.copy(url = it) },
                label = { Text("URL") },
                modifier = Modifier.fillMaxWidth()
            )
            TextField(
                value = SimpleDateFormat("yyyy-MM-dd").format(editedNews.date),
                onValueChange = { newValue ->
                    val parsedDate = SimpleDateFormat("yyyy-MM-dd").parse(newValue)
                    parsedDate?.let { editedNews = editedNews.copy(date = it) }
                },
                label = { Text("Date") },
                modifier = Modifier.fillMaxWidth()
            )
            TextField(
                value = editedNews.content,
                onValueChange = { editedNews = editedNews.copy(content = it) },
                label = { Text("Content") },
                modifier = Modifier.fillMaxWidth()
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = {
                        // Save button clicked
                        sendPost(editedNews.toString())
                        isEditing = false
                    },
                    modifier = Modifier.padding(end = 8.dp)
                ) {
                    Text("Save")
                }
                Button(
                    onClick = { isEditing = false },
                    modifier = Modifier.padding(end = 8.dp)
                ) {
                    Text("Cancel")
                }
                Button(
                    onClick = { onRemoveClicked() },
                    colors = ButtonDefaults.buttonColors(backgroundColor = Color.Red),
                    modifier = Modifier.padding(end = 8.dp)
                ) {
                    Text("Remove", color = Color.White)
                }
            }
        }
    } else {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = "Title: ${news.title}", style = MaterialTheme.typography.subtitle1)
            Text(text = "URL: ${news.url}", style = MaterialTheme.typography.subtitle1)
            Text(
                text = "Date: ${SimpleDateFormat("yyyy-MM-dd").format(news.date)}",
                style = MaterialTheme.typography.subtitle1
            )
            Text(text = "Content: ${news.content}", style = MaterialTheme.typography.subtitle1)

            Button(
                onClick = { isEditing = true },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(backgroundColor = Color.LightGray)
            ) {
                Text("Edit")
            }
        }
    }
}



@Composable
fun Main(selected: Section, news: ArrayList<INews>) {
    Surface(color = MaterialTheme.colors.background) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text(
                text = when (selected) {
                    Section.AddNews -> "${selected.name.uppercase(Locale.getDefault())} "
                    Section.DataBaseCollection -> "${selected.name.uppercase(Locale.getDefault())}"
                    Section.Scrape -> "${selected.name.uppercase(Locale.getDefault())}"

                },
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(bottom = 16.dp),
                style = MaterialTheme.typography.h6
            )

            if (selected == Section.DataBaseCollection) {
                val newsList = remember { mutableStateListOf(*news.toTypedArray()) }
                LazyColumn(
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(bottom = 16.dp)
                ) {
                    items(newsList) { item ->
                        var isEditing by remember { mutableStateOf(false) }
                        var editedNews by remember { mutableStateOf(item) }

                        if (isEditing) {
                            // Show edit fields
                            Column(
                                modifier = Modifier.padding(16.dp)
                            ) {
                                TextField(
                                    value = editedNews.title,
                                    onValueChange = { editedNews = editedNews.copy(title = it) },
                                    label = { Text("Title") }
                                )
                                TextField(
                                    value = editedNews.url,
                                    onValueChange = { editedNews = editedNews.copy(url = it) },
                                    label = { Text("URL") }
                                )
                                TextField(
                                    value = SimpleDateFormat("yyyy-MM-dd").format(editedNews.date),
                                    onValueChange = { newValue ->
                                        val parsedDate = SimpleDateFormat("yyyy-MM-dd").parse(newValue)
                                        parsedDate?.let { editedNews = editedNews.copy(date = it) }
                                    },
                                    label = { Text("Date") }
                                )
                                TextField(
                                    value = editedNews.content,
                                    onValueChange = { editedNews = editedNews.copy(content = it) },
                                    label = { Text("Content") }
                                )

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.End
                                ) {
                                    Button(
                                        onClick = {
                                            updateNews(item._id, editedNews.toString())
                                            //println(editedNews.toString())
                                            item.title = editedNews.title
                                            item.url = editedNews.url
                                            item.date = editedNews.date
                                            item.content = editedNews.content

                                            isEditing = false
                                        },
                                        modifier = Modifier.padding(end = 8.dp)
                                    ) {
                                        Text("Save")
                                    }
                                    Button(
                                        onClick = { isEditing = false }
                                    ) {
                                        Text("Cancel")
                                    }
                                }
                            }
                        } else {
                            NewsRow(
                                news = item,
                                onDeleteClicked = {
                                    //println(item._id)
                                    deleteNews(item._id)
                                    news.remove(item)
                                    newsList.remove(item)
                                },
                                onEditClicked = {
                                    isEditing = true
                                }
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))


            } else if (selected == Section.AddNews){
                val title = remember { mutableStateOf("") }
                val url = remember { mutableStateOf("") }
                val authors = remember { mutableStateOf("") }
                val categories = remember { mutableStateOf("") }
                val date = remember { mutableStateOf(Date()) }
                val content = remember { mutableStateOf("") }
                val newsList = remember { mutableStateListOf(*news.toTypedArray()) }

                Column(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Add New News", style = MaterialTheme.typography.h6)

                    TextField(
                        value = title.value,
                        onValueChange = { title.value = it },
                        label = { Text("Title") },
                        modifier = Modifier.fillMaxWidth()
                    )
                    TextField(
                        value = url.value,
                        onValueChange = { url.value = it },
                        label = { Text("URL") },
                        modifier = Modifier.fillMaxWidth()
                    )
                    TextField(
                        value = authors.value,
                        onValueChange = { authors.value = it },
                        label = { Text("Authors") },
                        modifier = Modifier.fillMaxWidth()
                    )
                    TextField(
                        value = categories.value,
                        onValueChange = { categories.value = it },
                        label = { Text("Categories") },
                        modifier = Modifier.fillMaxWidth()
                    )

                    TextField(
                        value = SimpleDateFormat("yyyy-MM-dd").format(date.value),
                        onValueChange = { newValue ->
                            val parsedDate = SimpleDateFormat("yyyy-MM-dd").parse(newValue)
                            parsedDate?.let { date.value = it }
                        },
                        label = { Text("Date") },
                        modifier = Modifier.fillMaxWidth()
                    )
                    TextField(
                        value = content.value,
                        onValueChange = { content.value = it },
                        label = { Text("Content") },
                        modifier = Modifier.fillMaxWidth()
                    )

                    Button(
                        onClick = {

                            val coords: Pair<Double, Double> = Pair(0.0, 0.0)

                            val newNews = INews(
                                title = title.value,
                                url = url.value,
                                date = date.value,
                                content = content.value,
                                authors = authors.value.split(",").map { it.trim() },
                                categories = categories.value.split(",").map { it.trim() },
                                location = Location(
                                    type = "Point",
                                    coordinates = coords,
                                )
                            )
                            //println(newNews.toString())
                            news.add(newNews)
                            newsList.add(newNews)
                            sendPost(newNews.toString())

                            title.value = ""
                            url.value = ""
                            date.value = Date()
                            content.value = ""
                            authors.value = ""
                            categories.value = ""

                        },
                        modifier = Modifier.align(Alignment.End)
                    ) {
                        Text("Post")
                    }
                }

            }else if (selected == Section.Scrape) {
                ScrapeSection()
            }
        }
    }
}

@Composable
fun App(news: ArrayList<INews>) {
    var selectedSection by remember { mutableStateOf(Section.DataBaseCollection) }

    MaterialTheme {
        Scaffold(
            content = { Main(selectedSection, news) },
            topBar = { Header { section -> selectedSection = section } },
            bottomBar = {
                Footer(
                    selected = selectedSection,
                    onFilterClicked = { /* Handle filter button click */ },
                    onOrderByClicked = { /* Handle order by button click */ }
                )
            }
        )
    }
}
