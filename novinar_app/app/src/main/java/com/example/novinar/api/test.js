const express = require("express");
const multer = require("multer");
const fs = require("fs");
const { MongoClient, ObjectId } = require("mongodb");

const app = express();
const port = 3000;

// MongoDB connection URI
const mongoUri = "mongodb+srv://ognjen:ognjen123@cluster0.allnp.mongodb.net/novinar?retryWrites=true&w=majority";

// Initialize MongoClient
const client = new MongoClient(mongoUri, { useNewUrlParser: true, useUnifiedTopology: true,    tlsAllowInvalidCertificates: true});

// Multer setup for file uploads
const upload = multer({ dest: "uploads/" });

app.use(express.json());

(async function startServer() {
    try {
        console.log("Connecting to MongoDB...");
        await client.connect();
        console.log("Connected to MongoDB successfully!");

        const db = client.db("novinar");
        const collection = db.collection("novice");

        // Predefined categories
        const categories = [
            "Politics",
            "Business",
            "Technology",
            "Sports",
            "Entertainment",
            "Health",
            "Science",
            "World",
            "Lifestyle",
            "Environment"
        ];

        // Root endpoint
        app.get("/", (req, res) => {
            res.send("Server is running and connected to MongoDB!");
        });

        // Endpoint to fetch all news
        app.get("/getNews", async (req, res) => {
            try {
                const results = await collection.find().toArray();
                res.status(200).send(results);
            } catch (error) {
                console.error("Error fetching news:", error);
                res.status(500).send({ error: "Failed to fetch news" });
            }
        });

        // Endpoint to fetch a single news item by ID
        app.get("/getNews/:id", async (req, res) => {
            try {
                const id = req.params.id;

                if (!ObjectId.isValid(id)) {
                    return res.status(400).send({ error: "Invalid ID format" });
                }

                const news = await collection.findOne({ _id: new ObjectId(id) });

                if (!news) {
                    return res.status(404).send({ error: "News not found" });
                }

                res.status(200).send(news);
            } catch (error) {
                console.error("Error fetching news by ID:", error);
                res.status(500).send({ error: "Failed to fetch news" });
            }
        });

        // Endpoint to add news
       // Endpoint to add news (image optional)
       app.post('/addNews', upload.single('image'), async (req, res) => {
        try {
            const { title, content, category, latitude, longitude } = req.body;
    
            // Validate input fields
            if (!title || !content || !category) {
                return res.status(400).send({ error: 'Title, content, and category are required.' });
            }
    
            let imageData = null;
    
            if (req.body.image === 'no_image') {
                imageData = 'no_image';
            } else if (req.file) {
                const imagePath = req.file.path;
                const imageBase64 = fs.readFileSync(imagePath, 'base64');
                imageData = `data:${req.file.mimetype};base64,${imageBase64}`;
                fs.unlinkSync(imagePath);
            }
    
            const news = {
                title,
                content,
                category,
                latitude: parseFloat(latitude) || 0.0,
                longitude: parseFloat(longitude) || 0.0,
                image: imageData,
                timestamp: new Date(),
            };
    
            const result = await collection.insertOne(news);
            res.status(201).send(result);
        } catch (error) {
            console.error('Error adding news:', error);
            res.status(500).send({ error: 'Failed to add news.' });
        }
    });
    


        // Endpoint to update news by ID
        app.put("/updateNews/:id", upload.single("image"), async (req, res) => {
            try {
                const id = req.params.id;

                if (!ObjectId.isValid(id)) {
                    return res.status(400).send({ error: "Invalid ID format" });
                }

                const { title, content, category } = req.body;
                const updatedData = {};

                if (title) updatedData.title = title;
                if (content) updatedData.content = content;
                if (category) {
                    if (!categories.includes(category)) {
                        return res.status(400).send({ error: "Invalid category" });
                    }
                    updatedData.category = category;
                }

                // Handle image update
                if (req.file) {
                    const imagePath = req.file.path;
                    const imageBase64 = fs.readFileSync(imagePath, "base64");
                    updatedData.image = `data:${req.file.mimetype};base64,${imageBase64}`;
                    fs.unlinkSync(imagePath);
                }

                const result = await collection.updateOne(
                    { _id: new ObjectId(id) },
                    { $set: updatedData }
                );

                if (result.matchedCount === 0) {
                    return res.status(404).send({ error: "News not found" });
                }

                res.status(200).send({ message: "News updated successfully", result });
            } catch (error) {
                console.error("Error updating news:", error);
                res.status(500).send({ error: "Failed to update news" });
            }
        });

        // Endpoint to delete news by ID
        app.delete("/deleteNews/:id", async (req, res) => {
            try {
                const id = req.params.id;

                if (!ObjectId.isValid(id)) {
                    return res.status(400).send({ error: "Invalid ID format" });
                }

                const result = await collection.deleteOne({ _id: new ObjectId(id) });

                if (result.deletedCount === 0) {
                    return res.status(404).send({ error: "News not found" });
                }

                res.status(200).send({ message: "News deleted successfully", result });
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
        process.exit(1);
    }
})();
