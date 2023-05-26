import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.*
import org.example.model.INews


enum class Section {
    Invoices,
    About
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
                        onSelectionChanged(Section.Invoices)
                    }
                    .weight(0.5f),
                horizontalArrangement = Arrangement.Center
            ) {
                Icon(
                    imageVector = Icons.Default.List,
                    contentDescription = "Invoices",
                    tint = Color.White,
                    modifier = Modifier
                        .size(24.dp)
                        .padding(end = 8.dp),
                )
                Text(
                    text = "Invoices",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                )
            }
            Row(
                modifier = Modifier
                    .clickable {
                        onSelectionChanged(Section.About)
                    }
                    .weight(0.5f),
                horizontalArrangement = Arrangement.Center

            ) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = "About",
                    tint = Color.White,
                    modifier = Modifier
                        .size(24.dp)
                        .padding(end = 8.dp)
                )
                Text(
                    text = "About",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}

@Composable
fun Footer(selected: Section) {
    BottomAppBar(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colors.primary)
    ) {
        Text(
            text = "You're viewing the ${selected.name} tab",
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
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
fun Main(selected: Section, news: ArrayList<INews>) {
    Surface(color = MaterialTheme.colors.background) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text(
                text = when (selected) {
                    Section.About -> "${selected.name.uppercase(Locale.getDefault())} CONTENT\n\n Subject: Principles of programming languages\nAuthor: Ognjen Vučković"
                    Section.Invoices -> "${selected.name.uppercase(Locale.getDefault())}"
                },
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(bottom = 16.dp),
                style = MaterialTheme.typography.h6
            )

            if (selected == Section.Invoices) {
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
                                        // Parse the date string and update the editedNews.date accordingly
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

                                // Save and cancel buttons
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.End
                                ) {
                                    Button(
                                        onClick = {
                                            // Perform the necessary validations and updates
                                            item.title = editedNews.title
                                            item.url = editedNews.url
                                            item.date = editedNews.date
                                            item.content = editedNews.content

                                            // Exit edit mode
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
                            // Show news item
                            NewsRow(
                                news = item,
                                onDeleteClicked = {
                                    newsList.removeAll { it.url == item.url }
                                },
                                onEditClicked = {
                                    // Enter edit mode
                                    isEditing = true
                                }
                            )
                        }
                    }
                }


            }
        }
    }
}
@Composable
fun App(onSelectionChanged: (Section) -> Unit, news: ArrayList<INews>) {
    var selectedSection by remember { mutableStateOf(Section.Invoices) }

    MaterialTheme {
        Scaffold(
            content = { Main(selectedSection, news) },
            topBar = { Header { section -> selectedSection = section } },
            bottomBar = { Footer(selectedSection) }
        )
    }
}

