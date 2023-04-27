import express, { Express, Request, Response } from 'express';
import { websites } from './scraper/websites';
import dotenv from 'dotenv';
import cheerio from 'cheerio';
import axios from 'axios';
import mongoose from 'mongoose';
import { INews } from './model/News';

dotenv.config();

const app = express();
app.use(express.json());

app.use('/api/news', require('./routes/news'));

app.get('/', async (req: Request, res: Response) => {
  mongoose.connect(
    `mongodb+srv://${process.env.DB_USERNAME}:${process.env.DB_PASSWORD}@cluster0.bd0tfwp.mongodb.net/?retryWrites=true&w=majority`
  );

  let news: INews[] = [];

  for await (let [key, value] of websites) {
    const valueResult = await value(5);
    for (let value of valueResult) {
      news.push(value);
    }
  }

  res.send(news);
});

app.listen(process.env.PORT, () => {
  console.log(
    `⚡️[server]: Server is running at http://localhost:${process.env.PORT}`
  );
});
