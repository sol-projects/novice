import express, { Express, Request, Response } from 'express'
import { websites } from './config/websites'
import dotenv from 'dotenv'
import cheerio from 'cheerio'
import axios from 'axios'

dotenv.config()

const app = express()
app.use(express.json())

app.use("/api/news", require('./routes/news'))

app.get("/", async (req: Request, res: Response) => {

    for await (let [key, value] of websites) {
        console.log(`Key: ${key}, Value: ${value()}`);
    }

    res.send("get news")
})

app.listen(process.env.PORT, () => {
  console.log(`⚡️[server]: Server is running at http://localhost:${process.env.PORT}`)
})
