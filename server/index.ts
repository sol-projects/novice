import express, { Express, Request, Response } from 'express';
import websites from './scraper/websites';
import dotenv from 'dotenv';
import cheerio from 'cheerio';
import axios from 'axios';
import mongoose from 'mongoose';
import { INews } from './model/News';
import router from './routes/news';

dotenv.config();

const app = express();
app.use(express.json());
app.use('/news', router);

mongoose.connect(
  `mongodb+srv://${process.env.DB_USERNAME}:${process.env.DB_PASSWORD}@cluster0.bd0tfwp.mongodb.net/?retryWrites=true&w=majority`
);

app.get('/', async (req: Request, res: Response) => {
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
  const site = `http://localhost:${process.env.PORT}`;
  console.log(
    `List of routes:
        GET:
        ${site}/news/
        ${site}/news/:id
        ${site}/news/scrape/:website/:n
        ${site}/news/categories/:categories
        ${site}/news/authors/:authors
        ${site}/news/location/:location
        ${site}/news/website/:website
        ${site}/news/date/before/:date
        ${site}/news/date/after/:date
        ${site}/news/date/after/:after/before/:before
        ${site}/news/title/:title
        ${site}/news/content/:content
        POST:
        ${site}/news/`
  );
});
