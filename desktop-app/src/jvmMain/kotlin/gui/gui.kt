
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.List
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import java.util.*

enum class Section {
    Invoices,
    About
}

@Composable
fun Header(onSelectionChanged: (Section) -> Unit) {
    var selected by remember { mutableStateOf(Section.Invoices) }

    TopAppBar(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colors.primary)
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 16.dp)
                .fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row (
                modifier = Modifier
                    .clickable{
                        if (selected != Section.Invoices) {
                            selected = Section.Invoices
                            onSelectionChanged(selected)
                        }
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
                    .clickable{
                        if (selected != Section.About) {
                            selected = Section.About
                            onSelectionChanged(selected)
                        }
                    }
                    .weight(0.5f),
                horizontalArrangement = Arrangement.Center

            ){
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
fun Main(selected: Section) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color.White)
    ) {
        Text(
            text = when(selected) {
                Section.About -> "\n${selected.name.uppercase(Locale.getDefault())} CONTENT\n\n Subject: Principles of programming languages\nAuthor: Liam Mesarec"
                Section.Invoices -> "\n${selected.name.uppercase(Locale.getDefault())} CONTENT\n\n"
            },
            textAlign = TextAlign.Center,
            modifier = Modifier
                .align(Alignment.Center)
        )
    }
}

@Composable
fun App() {
    var selected by remember { mutableStateOf(Section.Invoices) }
    MaterialTheme {
        Scaffold(
            content = {Main(selected)},
            topBar = {Header(onSelectionChanged = { selected = it }) },
            bottomBar = {Footer(selected)}
        )
    }
}
