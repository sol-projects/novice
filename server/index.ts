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
const cors = require('cors');

dotenv.config();

const app: Express = express();
app.use(cors({ credentials: true }));
app.use(express.json());
app.use('/news', router);

if (!process.env.DB_NAME) {
  console.error(`DB with name ${!process.env.DB_NAME} not found`);
} else {
  Db.connect(process.env.DB_NAME);
}

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

export default app;
