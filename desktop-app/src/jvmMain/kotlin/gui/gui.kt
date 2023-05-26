import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.List
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.sql.SQLException
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
            Row(
                modifier = Modifier
                    .clickable {
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
                    .clickable {
                        if (selected != Section.About) {
                            selected = Section.About
                            onSelectionChanged(selected)
                        }
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
fun InvoiceRow(invoice: Item) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 8.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = invoice.name,
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.subtitle1
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Price: $${invoice.price}",
                    style = MaterialTheme.typography.body1
                )
                Text(
                    text = "Quantity: ${invoice.quantity}",
                    style = MaterialTheme.typography.body1
                )
                Text(
                    text = "Barcode: ${invoice.barcode}",
                    style = MaterialTheme.typography.body1
                )
            }
            IconButton(
                onClick = { deleteItemFromDatabase(invoice) }
            ) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = "Delete",
                    tint = Color.Red
                )
            }
        }
    }
}

@Composable
fun Main(selected: Section, items: List<Item>) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color.White)
    ) {
        Column(
            modifier = Modifier
                .align(Alignment.Center)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Text(
                text = when (selected) {
                    Section.About -> "\n${selected.name.uppercase(Locale.getDefault())} CONTENT\n\n Subject: Principles of programming languages\nAuthor: Ognjen Vučković"
                    Section.Invoices -> "\n${selected.name.uppercase(Locale.getDefault())} CONTENT\n\n"
                },
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(bottom = 16.dp),
                style = MaterialTheme.typography.h6
            )

            if (selected == Section.Invoices) {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(bottom = 16.dp)
                ) {
                    items(items = items) { item ->
                        InvoiceRow(invoice = item)
                    }
                }
            }
        }
    }
}

@Composable
fun App() {
    var selected by remember { mutableStateOf(Section.Invoices) }

    val items = retrieveItemsFromDatabase()

    MaterialTheme {
        Scaffold(
            content = { Main(selected, items) },
            topBar = { Header(onSelectionChanged = { selected = it }) },
            bottomBar = { Footer(selected) }
        )
    }
}

fun retrieveItemsFromDatabase(): List<Item> {
    val connection = DatabaseUtil.getConnection()
    val items = mutableListOf<Item>()

    try {
        val statement = connection.createStatement()
        val selectQuery = "SELECT * FROM item"

        try {
            val resultSet = statement.executeQuery(selectQuery)

            while (resultSet.next()) {
                val name = resultSet.getString("name")
                val price = resultSet.getInt("price")
                val quantity = resultSet.getInt("quantity")
                val tax = resultSet.getDouble("tax")
                val barcode = resultSet.getString("barcode")

                val item = Item(name, price, quantity, tax, barcode)

                items.add(item)
            }
        } catch (e: SQLException) {
            println("Error executing SQL query: ${e.message}")
        } finally {
            statement.close()
        }
    } catch (e: SQLException) {
        println("Error establishing database connection: ${e.message}")
    } finally {
        connection.close()
    }

    return items
}

fun deleteItemFromDatabase(item: Item) {
    val connection = DatabaseUtil.getConnection()
    println(item.id)
    try {
        val statement = connection.createStatement()
        val deleteQuery = "DELETE FROM item WHERE id = '${item.id}'"

        try {
            val rowsAffected = statement.executeUpdate(deleteQuery)
            if (rowsAffected > 0) {
                println("Item deleted successfully")
            } else {
                println("Item not found in the database")
            }
        } catch (e: SQLException) {
            println("Error executing SQL query: ${e.message}")
        } finally {
            statement.close()
        }
    } catch (e: SQLException) {
        println("Error establishing database connection: ${e.message}")
    } finally {
        connection.close()
    }
}
