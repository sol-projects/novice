import express, { Express, Request, Response } from 'express';
import websites from './scraper/websites';
import dotenv from 'dotenv';
import cheerio from 'cheerio';
import axios from 'axios';
import { INews } from './model/News';
import router from './routes/news';
const http = require('http');
import * as Db from './db/db';
import * as Socket from './socket/socket';

dotenv.config();

const app = express();
app.use(express.json());
app.use('/news', router);
Db.connect();

const server = http.createServer(app);
Socket.init(server);

const site = `http://localhost:${process.env.PORT}`;
const routes = {
  GET: [
    `${site}/news/`,
    `${site}/news/:id`,
    `${site}/news/scrape/:n`,
    `${site}/news/scrape/:website/:n`,
    `${site}/news/categories/:categories`,
    `${site}/news/authors/:authors`,
    `${site}/news/location/:location`,
    `${site}/news/website/:website`,
    `${site}/news/date/before/:date`,
    `${site}/news/date/after/:date`,
    `${site}/news/date/after/:after/before/:before`,
    `${site}/news/title/:title`,
    `${site}/news/content/:content`,
  ],
  POST: [`${site}/news/`],
  DELETE: [`${site}/news/:id`],
};

app.get('/', async (req: Request, res: Response) => {
  res.send(routes);
});

server.listen(process.env.PORT, async () => {
  console.log(`routes: ${JSON.stringify(routes, null, '\t')}`);
});
