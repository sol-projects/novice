import { INews } from '../../model/News';
import * as Db from '../../db/db';
import puppeteer from 'puppeteer';
import { parse } from 'date-fns';
//TODO :: scrape sub title and quot(figure)
async function rtvSlo(n: number) {
  const newsList: INews[] = [];

  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  await page.goto('https://www.rtvslo.si/novice');

  const links = await page.$$('div.md-news');

  for (let i = 0; i < links.length && newsList.length < n; i++) {
    const link = links[i];

    const url = await link.$eval('a', (anchor) =>
      anchor ? `https://www.rtvslo.si${anchor.getAttribute('href')}` : null
    );

    if (url) {
      const articlePage = await browser.newPage();
      await articlePage.goto(url);

      const title = await articlePage.$eval(
        'header.article-header h1',
        (element) => element?.textContent?.trim() || ''
      );

      const dateString = await articlePage.$eval(
        'meta[name="published_date"]',
        (el) => el?.getAttribute('content') || ''
      );
      const date = parse(dateString, "yyyy-MM-dd'T'HH:mm:ss'Z'", new Date());

      const contentTmp = await articlePage.$eval(
        'article.article',
        (element) => element.textContent?.trim() || ''
      );
      const content = contentTmp.replace(/[\t\n]/g, '').trim();

      const metaElements = await articlePage.$$('meta.elastic[name="author"]');
      const authors = await Promise.all(
        metaElements.map(async (el) => {
          const content = await el.getProperty('content');
          const author = await content.jsonValue();
          return author ?? '';
        })
      );

      const locationTmp = await articlePage.$eval(
        'div.place-source',
        (el) => (el as HTMLElement).textContent?.trim() || ''
      );

      const coords: [number, number] =
        locationTmp !== 'MMC RTV SLO' && locationTmp !== ''
          ? await Db.Util.toCoords(locationTmp.split(', ')[0])
          : [0, 0];

      const categoriesString = await articlePage.$eval(
        'meta[name="keywords"]',
        (el) => el?.getAttribute('content')?.toLowerCase() || ''
      );
      const categories = categoriesString ? categoriesString.split(',') : [];

      const news: INews = {
        title,
        url,
        date,
        authors,
        content,
        categories,
        views: [],
        location: {
          type: 'Point',
          coordinates: coords,
        },
      };
      newsList.push(news);

      await articlePage.close();
    }
  }

  await browser.close();

  return newsList;
}

export = rtvSlo;
