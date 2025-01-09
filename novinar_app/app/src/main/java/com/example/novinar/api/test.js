const express = require("express");
const { MongoClient, ObjectId } = require("mongodb");

const app = express();
const port = 3000;

// MongoDB connection URI
const mongoUri = "";

// Initialize MongoClient
const client = new MongoClient(mongoUri, { useNewUrlParser: true, useUnifiedTopology: true });

app.use(express.json());

(async function startServer() {
    try {
        console.log("Connecting to MongoDB...");
        await client.connect(); // Explicitly await connection
        console.log("Connected to MongoDB successfully!");

        const db = client.db("novinar");
        const collection = db.collection("novice");

        // Root endpoint
        app.get("/", (req, res) => {
            res.send("Server is running and connected to MongoDB!");
        });

        // Endpoint to fetch all news
        app.get("/getNews", async (req, res) => {
            try {
                console.log("Fetching news from MongoDB...");
                const results = await collection.find().toArray();
                console.log("Fetched news:", results);
                res.status(200).send(results);
            } catch (error) {
                console.error("Error fetching news:", error);
                res.status(500).send({ error: "Failed to fetch news" });
            }
        });

        // Endpoint to add news
        app.post("/addNews", async (req, res) => {
            try {
                const news = req.body;
                news.timestamp = new Date(); // Add a timestamp
                const result = await collection.insertOne(news);
                console.log("Added news:", result);
                res.status(200).send(result);
            } catch (error) {
                console.error("Error adding news:", error);
                res.status(500).send({ error: "Failed to add news" });
            }
        });

        // Endpoint to update news by ID
        app.put("/updateNews/:id", async (req, res) => {
            try {
                const id = new ObjectId(req.params.id);
                const updatedData = req.body;
                const result = await collection.updateOne({ _id: id }, { $set: updatedData });
                console.log("Updated news:", result);
                res.status(200).send(result);
            } catch (error) {
                console.error("Error updating news:", error);
                res.status(500).send({ error: "Failed to update news" });
            }
        });

        // Endpoint to delete news by ID
        app.delete("/deleteNews/:id", async (req, res) => {
            try {
                const id = new ObjectId(req.params.id);
                const result = await collection.deleteOne({ _id: id });
                console.log("Deleted news:", result);
                res.status(200).send(result);
            } catch (error) {
                console.error("Error deleting news:", error);
                res.status(500).send({ error: "Failed to delete news" });
            }
        });

        // Start the Express server
        app.listen(port, () => {
            console.log(`Server running on http://localhost:${port}`);
        });
    } catch (error) {
        console.error("Error connecting to MongoDB:", error);
        process.exit(1); // Exit if MongoDB connection fails
    }
})();
